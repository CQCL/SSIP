# Functions for constructing the two pushouts necessary
# for a fault-tolerant code merge between CSS codes.
import numpy as np

from ssip.basic_functions import (
    BinMatrix,
    CSScode,
    direct_sum_matrices,
    find_homology_basis,
    find_paired_basis,
    num_data_qubits,
    num_logical_qubits,
)
from ssip.lifted_product import tensor_product
from ssip.merge_result import MergeResult, find_logical_splitting
from ssip.monic_span_checker import monic_span, restricted_matrix


# Calculates the tensor product in the category of chain complexes
# of a logical operator subcomplex V and the complex P1 -> P0,
# with delP: P1 -> P0 = (1 1)^T.
# See p32 of https://arxiv.org/abs/2301.13738.
def sandwich_middle_code(V: BinMatrix, depth: int = 1) -> CSScode:
    def gen_classical_code(depth):
        parity_mat = np.zeros((depth + 1, depth))
        for i in range(depth):
            parity_mat[i][i] = 1
            parity_mat[i + 1][i] = 1
        return parity_mat

    A = gen_classical_code(depth)
    return tensor_product(A, V)


# Calculates the pushout of a basis-preserving monic span with
# logical operator subcomplex as
# the apex. Assumes we have A, B differentials belonging
# to one complex and C, D to another.
def pushout(
    A: BinMatrix,
    B: BinMatrix,
    C: BinMatrix,
    D: BinMatrix,
    qubit_map: dict,
    syndrome_map: dict,
    return_data: bool = False,
) -> CSScode | MergeResult:
    # the first differential
    direct_sumAC = direct_sum_matrices(A, C)

    # quotient rows together
    for item in qubit_map.items():
        direct_sumAC[item[0]] = np.array(
            list(
                map(
                    lambda x, y: bool(x + y),
                    direct_sumAC[item[0]],
                    direct_sumAC[item[1] + A.shape[0]],
                )
            )
        )

    del2 = np.delete(direct_sumAC, [x + A.shape[0] for x in qubit_map.values()], 0)
    # the second differential
    direct_sumBD = direct_sum_matrices(B, D)

    # quotient rows together
    for item in syndrome_map.items():
        direct_sumBD[item[0]] = np.array(
            list(
                map(
                    lambda x, y: bool(x + y),
                    direct_sumBD[item[0]],
                    direct_sumBD[item[1] + B.shape[0]],
                )
            )
        )

    del1 = np.delete(direct_sumBD, [x + B.shape[0] for x in syndrome_map.values()], 0)

    # quotient columns together
    for item in qubit_map.items():
        del1[:, item[0]] = np.array(
            list(
                map(
                    lambda x, y: bool(x + y),
                    del1[:, item[0]],
                    del1[:, item[1] + B.shape[1]],
                )
            )
        )
    del1 = np.delete(del1, [x + B.shape[1] for x in qubit_map.values()], 1)

    new_code = CSScode(del2.T, del1)

    if return_data:
        # Calculate coequaliser at degree 1
        n = A.shape[0] + C.shape[0]
        n2 = A.shape[0]
        coeq = np.eye(n)
        for key, val in qubit_map.items():
            coeq[key] = np.array(
                list(map(lambda x, y: bool(x + y), coeq[key], coeq[val + n2]))
            )
        coeq = np.delete(coeq, [x + n2 for x in qubit_map.values()], 0)

        # Calculate whether there are new logical qubits introduced by the merge
        num_before_logicals = len(find_homology_basis(A, B)) + len(
            find_homology_basis(C, D)
        )
        num_after_logicals = num_logical_qubits(new_code)
        num_new_logicals = num_after_logicals - num_before_logicals + 1

        new_Z_logicals = []
        new_X_logicals = []

        old_Z_logicals = []
        old_X_logicals = []

        if num_new_logicals < 0:
            raise ValueError("Pushout error, logical operators are not irreducible")
        # Calculate paired Z/X logicals
        if num_new_logicals:
            old_code = CSScode(direct_sum_matrices(A, C).T, direct_sum_matrices(B, D))
            new_Z_logicals, new_X_logicals, old_Z_logicals, old_X_logicals = (
                find_logical_splitting(
                    old_code, new_code, coeq, num_before_logicals, num_after_logicals
                )
            )

        else:
            old_Z_logicals = find_homology_basis(del2, del1)
            trial_X_basis = find_homology_basis(del1.T, del2.T)
            old_X_logicals = find_paired_basis(old_Z_logicals, trial_X_basis)

        return MergeResult(
            new_code,
            coeq,
            [],
            [],
            [],
            new_Z_logicals,
            new_X_logicals,
            old_Z_logicals,
            old_X_logicals,
        )

    return new_code


# Calculates the pushout based on qubits labelled by indices; if these indices do not
# yield a monic span then return None. Assume C, D are CSScodes.
def pushout_by_indices(
    C: CSScode,
    D: CSScode,
    indices1: list[int],
    indices2: list[int],
    basis: str = "Z",
    return_data: bool = False,
) -> CSScode | MergeResult | None:
    restr1 = None
    restr2 = None
    if basis == "Z":
        restr1 = restricted_matrix(indices1, C.PX)
        restr2 = restricted_matrix(indices2, D.PX)
    elif basis == "X":
        restr1 = restricted_matrix(indices1, C.PZ)
        restr2 = restricted_matrix(indices2, D.PZ)
    else:
        raise ValueError("Must enter a valid Pauli string.")

    span = monic_span(restr1[0], restr2[0])
    if span is None:
        return None
    qubit_map = {}
    for qb1, qb2 in span[0].items():
        qubit_map[indices1[qb1]] = indices2[qb2]
    syndrome_map = {}
    for syndrome1, syndrome2 in span[1].items():
        syndrome_map[restr1[1][syndrome1]] = restr2[1][syndrome2]
    if basis == "Z":
        return pushout(
            C.PZ.T.copy(),
            C.PX.copy(),
            D.PZ.T.copy(),
            D.PX.copy(),
            qubit_map,
            syndrome_map,
            return_data,
        )
    elif basis == "X":
        temp_code = pushout(
            C.PX.T.copy(),
            C.PZ.copy(),
            D.PX.T.copy(),
            D.PZ.copy(),
            qubit_map,
            syndrome_map,
            return_data,
        )
        if return_data:
            temp_new_Z_logicals = temp_code.NewZLogicals.copy()
            temp_code.NewZLogicals = temp_code.NewXLogicals.copy()
            temp_code.NewXLogicals = temp_new_Z_logicals.copy()
            temp_old_Z_logicals = temp_code.OldZLogicals.copy()
            temp_code.OldZLogicals = temp_code.OldXLogicals.copy()
            temp_code.OldXLogicals = temp_old_Z_logicals.copy()
            temp_code.Code = CSScode(temp_code.Code.PX, temp_code.Code.PZ)
            return temp_code
        return CSScode(temp_code.PX, temp_code.PZ)


# Given two CSS codes C2 -> C1 -> C0 and D2 -> D1 -> C0, with F: C2 -> C1, G: C1 -> 0,
# J: D2 -> D1, K: D1 -> D0, and matching logical operators u in C1 and v in D1,
# constructs the pushout of a pushout of chain complexes, giving the final code.
# Assumes that the pushouts are of basis-preserving monic spans with logical operator
# subcomplexes at the apexes.
def external_merge(
    V: BinMatrix,
    F: BinMatrix,
    G: BinMatrix,
    J: BinMatrix,
    K: BinMatrix,
    qubit_map: dict,
    syndrome_map: dict,
    depth: int = 1,
    return_data: str = False,
) -> CSScode | MergeResult:
    if depth < 1:
        raise ValueError("Must enter a pushout depth greater than 0.")

    tens = sandwich_middle_code(V, depth)
    qmap1 = {}
    qmap2 = {}
    for count, item in enumerate(qubit_map.items()):
        qmap1[count] = item[0]
        qmap2[count + depth * len(qubit_map)] = item[1]
    smap1 = {}
    smap2 = {}
    for count, item in enumerate(syndrome_map.items()):
        smap1[count] = item[0]
        smap2[count + depth * len(syndrome_map)] = item[1]

    first_pushout = pushout(tens.PZ.T, tens.PX, F, G, qmap1, smap1)
    second_pushout = pushout(first_pushout.PZ.T, first_pushout.PX, J, K, qmap2, smap2)

    if return_data:
        n_before = G.shape[1] + K.shape[1]
        n_after = num_data_qubits(second_pushout)
        if not n_after == G.shape[1] + K.shape[1] + (depth - 1) * len(
            qubit_map
        ) + depth * len(syndrome_map):
            raise RuntimeError("Dimensions do not match up when returning merge data")

        new_qubits = list(np.arange(len(qubit_map), depth * len(qubit_map))) + list(
            np.arange(
                (depth + 1) * len(qubit_map),
                (depth + 1) * len(qubit_map) + depth * len(syndrome_map),
            )
        )
        new_Z_stabilisers = list(range(depth * len(qubit_map)))
        new_X_stabilisers = list(range(len(syndrome_map), depth * len(syndrome_map)))

        merge_map = np.zeros((n_after, n_before))

        qmap1_flipped = {v: k for k, v in qmap1.items()}
        qmap2_flipped = {v: k for k, v in qmap2.items()}

        count1 = 0
        count2 = 0

        for i in range(G.shape[1]):
            if i in qmap1_flipped:
                merge_map[qmap1_flipped[i]][i] = 1
                count1 += 1
            else:
                merge_map[
                    (depth + 1) * len(qubit_map)
                    + depth * len(syndrome_map)
                    + i
                    - count1
                ][i] = 1

        for i in range(K.shape[1]):
            if i in qmap2_flipped:
                merge_map[qmap2_flipped[i]][i + G.shape[1]] = 1
                count2 += 1
            else:
                j = (
                    G.shape[1]
                    + depth * len(qmap2)
                    + depth * len(syndrome_map)
                    + i
                    - count2
                )
                merge_map[j][i + G.shape[1]] = 1

        num_before_logicals = num_logical_qubits(CSScode(F.T, G)) + num_logical_qubits(
            CSScode(J.T, K)
        )
        num_after_logicals = num_logical_qubits(second_pushout)
        num_new_logicals = num_after_logicals - num_before_logicals + 1
        new_Z_logicals = []
        new_X_logicals = []

        old_Z_logicals = []
        old_X_logicals = []

        if num_new_logicals < 0:
            raise ValueError(
                "External merge error, logical operators are not irreducible"
            )
        if num_new_logicals:
            old_code = CSScode(direct_sum_matrices(F, J).T, direct_sum_matrices(G, K))
            new_Z_logicals, new_X_logicals, old_Z_logicals, old_X_logicals = (
                find_logical_splitting(
                    old_code,
                    second_pushout,
                    merge_map,
                    num_before_logicals,
                    num_after_logicals,
                )
            )
        else:
            old_Z_logicals = find_homology_basis(second_pushout.PZ.T, second_pushout.PX)
            trial_X_logicals = find_homology_basis(
                second_pushout.PX.T, second_pushout.PZ
            )
            old_X_logicals = find_paired_basis(old_Z_logicals, trial_X_logicals)

        return MergeResult(
            second_pushout,
            merge_map,
            new_Z_stabilisers,
            new_X_stabilisers,
            new_qubits,
            new_Z_logicals,
            new_X_logicals,
            old_Z_logicals,
            old_X_logicals,
        )

    return second_pushout


def external_merge_by_indices(
    C: CSScode,
    D: CSScode,
    indices1: list[int],
    indices2: list[int],
    basis: str = "Z",
    depth: int = 1,
    return_data: bool = False,
) -> CSScode | MergeResult | None:
    """Construct an external merge between two CSS codeblocks, along two
    logical operators.

    Args:
        C: The first CSS codeblock.
        D: The second CSS codeblock.
        indices1: The qubits in the support of the first logical operator.
        indices2: The qubits in the support of the second logical operator.
        basis: The basis, X or Z, to perform the logical measurement in.
        depth: The depth of the merge, i.e. how large to make the tensor product intermediate code.
        return_data: Whether to calculate a MergeResult rather than a CSScode output. The MergeResult includes substantially more data about the merge.


    Return:
        The merged CSScode, or the MergeResult object which contains the CSScode, or None if no merge could be found.
    """
    restr1 = None
    restr2 = None

    if basis == "Z":
        restr1 = restricted_matrix(indices1, C.PX)
        restr2 = restricted_matrix(indices2, D.PX)
    elif basis == "X":
        restr1 = restricted_matrix(indices1, C.PZ)
        restr2 = restricted_matrix(indices2, D.PZ)
    else:
        raise ValueError("Must enter a valid Pauli string.")

    span = monic_span(restr1[0], restr2[0])
    if span is None:
        return None

    qubit_map = {}
    for qb1, qb2 in span[0].items():
        qubit_map[indices1[qb1]] = indices2[qb2]
    syndrome_map = {}
    for syndrome1, syndrome2 in span[1].items():
        syndrome_map[restr1[1][syndrome1]] = restr2[1][syndrome2]
    qubit_map = dict(sorted(qubit_map.items()))
    syndrome_map = dict(sorted(syndrome_map.items()))
    if basis == "Z":
        return external_merge(
            restr1[0],
            C.PZ.T.copy(),
            C.PX.copy(),
            D.PZ.T.copy(),
            D.PX.copy(),
            qubit_map,
            syndrome_map,
            depth,
            return_data,
        )
    if basis == "X":
        temp_code = external_merge(
            restr1[0],
            C.PX.T.copy(),
            C.PZ.copy(),
            D.PX.T.copy(),
            D.PZ.copy(),
            qubit_map,
            syndrome_map,
            depth,
            return_data,
        )
        if return_data:
            temp_code.Code = CSScode(temp_code.Code.PX, temp_code.Code.PZ)
            temp_new_X_stabs = temp_code.NewXStabs.copy()
            temp_code.NewXStabs = temp_code.NewZStabs.copy()
            temp_code.NewZStabs = temp_new_X_stabs
            temp_new_Z_logicals = temp_code.NewZLogicals.copy()
            temp_code.NewZLogicals = temp_code.NewXLogicals.copy()
            temp_code.NewXLogicals = temp_new_Z_logicals.copy()
            temp_old_Z_logicals = temp_code.OldZLogicals.copy()
            temp_code.OldZLogicals = temp_code.OldXLogicals.copy()
            temp_code.OldXLogicals = temp_old_Z_logicals.copy()
            return temp_code
        return CSScode(temp_code.PX, temp_code.PZ)
