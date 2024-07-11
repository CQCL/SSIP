# Functions to perform "internal merges" on a code.
# Follows basically the same prescription as pushouts
# and is strictly more general.

# One can prove that these quotients on rows and columns are the correct ones
# using the quotient chain map.
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
from ssip.merge_result import MergeResult, find_logical_splitting
from ssip.monic_span_checker import monic_span, restricted_matrix
from ssip.pushouts import sandwich_middle_code


# Assumes that there are no qubits which appear twice in the qubit_map;
# same for checks in the syndrome_map.
# A and B are differentials.
def coequaliser(
    A: BinMatrix,
    B: BinMatrix,
    qubit_map: dict,
    syndrome_map: dict,
    return_data: bool = False,
) -> CSScode | MergeResult:
    # First differential
    # Quotient rows together
    A2 = A.copy()
    B2 = B.copy()
    for item in qubit_map.items():
        # XOR, as two qubits being quotiented can share a Z-check
        A2[item[0]] = np.array(
            list(map(lambda x, y: bool((x + y) % 2), A2[item[0]], A2[item[1]]))
        )

    del2 = np.delete(A2, list(qubit_map.values()), 0)

    # Quotient rows together. Also an XOR
    for item in syndrome_map.items():
        B2[item[0]] = np.array(
            list(map(lambda x, y: bool((x + y) % 2), B2[item[0]], B2[item[1]]))
        )

    del1 = np.delete(B2, list(syndrome_map.values()), 0)

    # Second differential
    # Quotient columns together. Just an OR this time.
    for item in qubit_map.items():
        del1[:, item[0]] = np.array(
            list(map(lambda x, y: bool(x + y), del1[:, item[0]], del1[:, item[1]]))
        )

    del1 = np.delete(del1, list(qubit_map.values()), 1)
    new_code = CSScode(del2.T, del1)
    if return_data:
        coeq = np.eye(A.shape[0])
        for key, val in qubit_map.items():
            coeq[key] = np.array(
                list(map(lambda x, y: bool(x + y), coeq[key], coeq[val]))
            )
        coeq = np.delete(coeq, list(qubit_map.values()), 0)

        num_before_logicals = len(find_homology_basis(A, B))
        num_after_logicals = num_logical_qubits(new_code)
        num_new_logicals = num_after_logicals - num_before_logicals + 1

        new_Z_logicals = []
        new_X_logicals = []

        old_Z_logicals = []
        old_X_logicals = []

        if num_new_logicals < 0:
            raise ValueError("Coequaliser error, logical operators are not irreducible")
        if num_new_logicals:
            old_code = CSScode(A.T, B)
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


def coequaliser_by_indices(
    C: CSScode,
    indices1: list,
    indices2: list,
    basis: str = "Z",
    return_data: bool = False,
) -> CSScode | MergeResult | None:
    restr1 = None
    restr2 = None
    if basis == "Z":
        (restr1, rows_to_keep1) = restricted_matrix(indices1, C.PX)
        (restr2, rows_to_keep2) = restricted_matrix(indices2, C.PX)
    elif basis == "X":
        (restr1, rows_to_keep1) = restricted_matrix(indices1, C.PZ)
        (restr2, rows_to_keep2) = restricted_matrix(indices2, C.PZ)
    else:
        raise ValueError("Must enter a valid Pauli string.")
    span = monic_span(restr1, restr2)

    if span is None:
        return None
    qubit_map = {}
    for qb1, qb2 in span[0].items():
        qubit_map[indices1[qb1]] = indices2[qb2]
    syndrome_map = {}
    for syndrome1, syndrome2 in span[1].items():
        syndrome_map[rows_to_keep1[syndrome1]] = rows_to_keep2[syndrome2]

    if basis == "Z":
        return coequaliser(
            C.PZ.T.copy(), C.PX.copy(), qubit_map, syndrome_map, return_data
        )
    elif basis == "X":
        temp_code = coequaliser(
            C.PX.T.copy(), C.PZ.copy(), qubit_map, syndrome_map, return_data
        )
        if return_data:
            temp_new_Z_logicals = temp_code.NewZLogicals
            temp_code.NewZLogicals = temp_code.NewXLogicals
            temp_code.NewXLogicals = temp_new_Z_logicals
            temp_old_Z_logicals = temp_code.OldZLogicals
            temp_code.OldZLogicals = temp_code.OldXLogicals
            temp_code.OldXLogicals = temp_old_Z_logicals
            temp_code.Code = CSScode(temp_code.Code.PX, temp_code.Code.PZ)
            return temp_code
        return CSScode(temp_code.PX, temp_code.PZ)


# V the logical operator subcomplex, C the base code. Does not work if
# there are duplicate qubits or syndromes. The duplicates in the syndrome
# can be resolved mathematically if they appear in the same entry,
# but I do not know how to code this right now.
def internal_merge(
    V: BinMatrix,
    C: CSScode,
    qubit_map: dict,
    syndrome_map: dict,
    depth: int = 1,
    return_data: str = False,
) -> CSScode | MergeResult:
    tens = sandwich_middle_code(V, depth)
    direct_sum1 = direct_sum_matrices(tens.PZ.T, C.PZ.T)
    direct_sum2 = direct_sum_matrices(tens.PX, C.PX)

    num_new_qubits = num_data_qubits(tens)
    num_new_X_checks = tens.PX.shape[0]

    qmap1 = {}
    qmap2 = {}
    for count, item in enumerate(qubit_map.items()):
        qmap1[item[0] + num_new_qubits] = count
        qmap2[item[1] + num_new_qubits - len(qubit_map)] = (depth - 1) * len(
            qubit_map
        ) + count

    smap1 = {}
    smap2 = {}
    for count, item in enumerate(syndrome_map.items()):
        smap1[item[0] + num_new_X_checks] = count
        smap2[item[1] + num_new_X_checks - len(syndrome_map)] = (depth - 1) * len(
            syndrome_map
        ) + count

    first_coeq = coequaliser(direct_sum1, direct_sum2, qmap1, smap1)
    second_coeq = coequaliser(first_coeq.PZ.T, first_coeq.PX, qmap2, smap2)

    if return_data:
        n_before = num_data_qubits(C)
        n_after = num_data_qubits(second_coeq)

        if not n_after == num_data_qubits(C) + (depth - 1) * len(
            qubit_map
        ) + depth * len(syndrome_map):
            raise RuntimeError("Dimensions do not match up when returning merge data")

        new_qubits = list(
            range((depth - 1) * len(qubit_map) + depth * len(syndrome_map))
        )

        merge_map = np.zeros((n_after, n_before))
        for i in range(n_before):
            merge_map[i + len(new_qubits)][i] = 1

        new_Z_stabilisers = list(range(depth * len(qubit_map)))
        new_X_stabilisers = list(range((depth - 1) * len(syndrome_map)))

        num_before_logicals = num_logical_qubits(C)
        num_after_logicals = num_logical_qubits(second_coeq)
        num_new_logicals = num_after_logicals - num_before_logicals + 1

        new_Z_logicals = []
        new_X_logicals = []

        old_Z_logicals = []
        old_X_logicals = []

        if num_new_logicals < 0:
            raise ValueError(
                "Internal merge error, logical operators are not irreducible"
            )
        if num_new_logicals:
            new_Z_logicals, new_X_logicals, old_Z_logicals, old_X_logicals = (
                find_logical_splitting(
                    C, second_coeq, merge_map, num_before_logicals, num_after_logicals
                )
            )
        else:
            old_Z_logicals = find_homology_basis(second_coeq.PZ.T, second_coeq.PX)
            old_X_logicals = find_homology_basis(second_coeq.PX.T, second_coeq.PZ)
            old_X_logicals = find_paired_basis(old_Z_logicals, old_X_logicals)

        return MergeResult(
            second_coeq,
            merge_map,
            new_Z_stabilisers,
            new_X_stabilisers,
            new_qubits,
            new_Z_logicals,
            new_X_logicals,
            old_Z_logicals,
            old_X_logicals,
        )

    return second_coeq


def internal_merge_by_indices(
    C: CSScode,
    indices1: list[int],
    indices2: list[int],
    basis: str = "Z",
    depth: int = 1,
    return_data: bool = False,
) -> CSScode | MergeResult | None:
    """Construct an internal merge within one CSS codeblock, along two
    disjoint logical operators.

    Args:
        C: The CSScode to merge within.
        indices1: The qubits in the support of the first logical operator.
        indices2: The qubits in the support of the second logical operator.
        basis: The basis, X or Z, to perform the logical measurement in.
        depth: The depth of the merge, i.e. how large to make the tensor product intermediate code.
        return_data: Whether to calculate a MergeResult rather than a CSScode output. The MergeResult includes substantially more data about the merge.


    Return:
        The merged CSScode, or the MergeResult object which contains the CSScode, or None if no merge can be found.
    """
    restr1 = None
    restr2 = None
    if basis == "Z":
        (restr1, rows_to_keep1) = restricted_matrix(indices1, C.PX)
        (restr2, rows_to_keep2) = restricted_matrix(indices2, C.PX)
    elif basis == "X":
        (restr1, rows_to_keep1) = restricted_matrix(indices1, C.PZ)
        (restr2, rows_to_keep2) = restricted_matrix(indices2, C.PZ)
    else:
        raise ValueError("Must enter a valid Pauli string.")
    span = monic_span(restr1, restr2)
    if span is None:
        return None
    qubit_map = {}
    for qb1, qb2 in span[0].items():
        qubit_map[indices1[qb1]] = indices2[qb2]
    syndrome_map = {}
    for syndrome1, syndrome2 in span[1].items():
        syndrome_map[rows_to_keep1[syndrome1]] = rows_to_keep2[syndrome2]
    qubit_map = dict(sorted(qubit_map.items()))
    syndrome_map = dict(sorted(syndrome_map.items()))

    for val in qubit_map.values():
        if val in qubit_map:
            return None
    # if there are duplicate syndromes do not merge, as it yields a
    # qubit with no X-checks, hence the code is distance 1
    for val in syndrome_map.values():
        if val in syndrome_map:
            return None

    if basis == "Z":
        return internal_merge(restr1, C, qubit_map, syndrome_map, depth, return_data)
    elif basis == "X":
        temp_code = internal_merge(
            restr1, CSScode(C.PX, C.PZ), qubit_map, syndrome_map, depth, return_data
        )
        if return_data:
            temp_code.Code = CSScode(temp_code.Code.PX, temp_code.Code.PZ)
            temp_new_X_stabs = temp_code.NewXStabs.copy()
            temp_code.NewXStabs = temp_code.NewZStabs.copy()
            temp_code.NewZStabs = temp_new_X_stabs
            temp_new_Z_logicals = temp_code.NewZLogicals
            temp_code.NewZLogicals = temp_code.NewXLogicals
            temp_code.NewXLogicals = temp_new_Z_logicals
            temp_old_Z_logicals = temp_code.OldZLogicals
            temp_code.OldZLogicals = temp_code.OldXLogicals
            temp_code.OldXLogicals = temp_old_Z_logicals
            return temp_code
        return CSScode(temp_code.PX, temp_code.PZ)
