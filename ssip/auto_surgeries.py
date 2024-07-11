import copy
from dataclasses import dataclass

import numpy as np

from ssip.basic_functions import (
    CSScode,
    direct_sum_codes,
    find_paired_basis,
    find_X_basis,
    find_Z_basis,
    indices_to_vector,
    multiply_F2,
    num_data_qubits,
    num_logical_qubits,
    num_X_stabs,
    num_Z_stabs,
    vector_to_indices,
)
from ssip.coequalisers import internal_merge_by_indices
from ssip.lifted_product import tensor_product
from ssip.merge_result import MergeResult, find_logical_splitting
from ssip.monic_span_checker import (
    restricted_matrix,
)
from ssip.pushouts import pushout


@dataclass
class MergeSequence:
    """Data that informs `parallel_internal_merges` and
    `parallel_external_merges` which merges to perform.

    Attributes:
        IndexSequence: Each merge takes in a pair of sets of indices, corresponding to the logicals to be merged. An `IndexSequence` is hence a list of these pairs of sets.
        DepthSequence: A list of depths to perform the merges at.

    """

    IndexSequence: list
    DepthSequence: list


@dataclass
class MeasureSequence:
    """Data that informs `parallel_single_logical_qubit_measurements`
    which measurements to perform.

    Attributes:
        LogicalList: Each measurement requires a set of indices, i.e. the logical to be measured. A LogicalList is a list of these.
        DepthSequence: A list of depths to perform the merges at.

    """

    LogicalList: list
    DepthSequence: list


# see https://arxiv.org/abs/2110.10794.
# return_data primarily useful for obtaining the new Z and X logicals.
def measure_single_logical_qubit(
    C: CSScode,
    indices: list[int],
    basis: str = "Z",
    depth: int = 1,
    return_data: bool = False,
) -> CSScode | MergeResult:
    """Takes a CSS code and measures a logical qubit in the manner of
    https://arxiv.org/abs/2110.10794, in either the Z or X basis.

    Args:
        C: The CSScode.
        indices: The indices of the logical operator to be measured out.
        basis: The basis of the measurement.
        depth: Controls the size of the ancilla code used for measurement.
        return_data: Whether to return a MergeResult or a CSScode object.


    Returns:
        The merged CSScode, or the MergeResult object which contains the CSScode.
    """

    # generates a classical code which, when used in a tensor product,
    # can be used to make codes to measure single logical qubits.
    def generate_classical_line_code(n):
        mat = np.eye(n)
        for i in range(n - 1):
            mat[i + 1][i] = 1
        return mat

    restr, rows_to_keep = (None, None)

    if basis == "Z":
        restr, rows_to_keep = restricted_matrix(indices, C.PX)
    elif basis == "X":
        restr, rows_to_keep = restricted_matrix(indices, C.PZ)
    else:
        raise ValueError("Must enter a valid Pauli string.")

    rows_to_keep = sorted(rows_to_keep)

    V = restr
    S = generate_classical_line_code(depth)
    tens = tensor_product(S, V)
    n_initial = num_data_qubits(C)

    qmap = {v: k for k, v in enumerate(indices)}
    smap = {v: k for k, v in enumerate(rows_to_keep)}

    if basis == "Z":
        temp_code = pushout(
            C.PZ.T.copy(),
            C.PX.copy(),
            tens.PZ.T.copy(),
            tens.PX.copy(),
            qmap,
            smap,
            return_data,
        )
        if return_data:
            temp_code.MergeMap = temp_code.MergeMap[:, :n_initial]
            temp_code.NewZStabs = list(
                range(num_Z_stabs(C), num_Z_stabs(temp_code.Code))
            )
            temp_code.NewXStabs = list(
                range(num_X_stabs(C), num_X_stabs(temp_code.Code))
            )
            temp_code.NewQubits = list(
                range(num_data_qubits(C), num_data_qubits(temp_code.Code))
            )
        return temp_code
    elif basis == "X":
        temp_code = pushout(
            C.PX.T.copy(),
            C.PZ.copy(),
            tens.PZ.T.copy(),
            tens.PX.copy(),
            qmap,
            smap,
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
            temp_code.MergeMap = temp_code.MergeMap[:, :n_initial]
            temp_code.NewZStabs = list(
                range(num_Z_stabs(C), num_Z_stabs(temp_code.Code))
            )
            temp_code.NewXStabs = list(
                range(num_X_stabs(C), num_X_stabs(temp_code.Code))
            )
            temp_code.NewQubits = list(
                range(num_data_qubits(C), num_data_qubits(temp_code.Code))
            )

            return temp_code
        return CSScode(temp_code.PX, temp_code.PZ)


def parallel_single_logical_qubit_measurements(
    C: CSScode,
    measure_sequence: MeasureSequence,
    basis: str = "Z",
    return_data: bool = False,
) -> CSScode | MergeResult:
    """Takes a CSS code and measures several logical qubits in the manner of
    https://arxiv.org/abs/2110.10794, in either the Z or X basis, in parallel.

    Args:
        C: The CSScode.
        measure_sequence: The measurements to be performed.
        basis: The basis of the measurements.
        return_data: Whether to return a MergeResult or a CSScode object.


    Returns:
        The merged CSScode, or the MergeResult object which contains the CSScode.
    """
    merge_result = None
    total_merge_map = None

    num_measures = len(measure_sequence.LogicalList)
    for i in range(num_measures):
        if i == 0:
            merge_result = measure_single_logical_qubit(
                C,
                measure_sequence.LogicalList[0],
                basis,
                measure_sequence.DepthSequence[0],
                True,
            )
            if merge_result is None:
                raise RuntimeError(
                    "Parallel logical measurements cannot be found for the given input"
                )

            total_merge_map = merge_result.MergeMap
        else:
            n = total_merge_map.shape[1]
            v = indices_to_vector(measure_sequence.LogicalList[i], n)
            new_v = vector_to_indices(
                list((multiply_F2(total_merge_map, np.array([v]).T).T)[0])
            )

            merge_result = measure_single_logical_qubit(
                merge_result.Code,
                new_v,
                basis,
                measure_sequence.DepthSequence[i],
                True,
            )

            if merge_result is None:
                raise ValueError(
                    "Parallel logical measurements cannot be found for the given input"
                )
            total_merge_map = multiply_F2(merge_result.MergeMap, total_merge_map)

    if return_data:
        num_before_logicals = num_logical_qubits(C)
        num_after_logicals = num_logical_qubits(merge_result.Code)
        new_Z_logicals, new_X_logicals, old_Z_logicals, old_X_logicals = (
            find_logical_splitting(
                C,
                merge_result.Code,
                total_merge_map,
                num_before_logicals,
                num_after_logicals,
                num_measures,
                basis,
            )
        )
        new_Z_stabs = list(range(num_Z_stabs(C), num_Z_stabs(merge_result.Code)))
        new_X_stabs = list(range(num_X_stabs(C), num_X_stabs(merge_result.Code)))
        new_qubits = list(range(num_data_qubits(C), num_data_qubits(merge_result.Code)))

        return MergeResult(
            merge_result.Code,
            total_merge_map,
            new_Z_stabs,
            new_X_stabs,
            new_qubits,
            new_Z_logicals,
            new_X_logicals,
            old_Z_logicals,
            old_X_logicals,
        )

    return merge_result.Code


def parallel_internal_merges(
    C: CSScode,
    merge_sequence: MergeSequence,
    basis: str = "Z",
    return_data: bool = False,
) -> CSScode | MergeResult:
    """Takes a CSS code and performs several internal merges in parallel.

    Args:
        C: The CSScode.
        merge_sequence: The merges to be performed.
        basis: The basis of the measurements.
        return_data: Whether to return a MergeResult or a CSScode object.


    Returns:
        The merged CSScode, or the MergeResult object which contains the CSScode.
    """
    merge_result = None
    total_merge_map = None
    total_new_X_stabs = 0
    total_new_Z_stabs = 0
    num_new_qubits = 0
    num_merges = len(merge_sequence.IndexSequence)
    for i in range(num_merges):
        if i == 0:
            merge_result = internal_merge_by_indices(
                C,
                merge_sequence.IndexSequence[i][0],
                merge_sequence.IndexSequence[i][1],
                basis,
                merge_sequence.DepthSequence[i],
                True,
            )
            if merge_result is None:
                raise ValueError(
                    "parallel internal merges cannot be found for the given input"
                )

            total_merge_map = merge_result.MergeMap
            if merge_result.NewQubits is not None:
                num_new_qubits += len(merge_result.NewQubits)
            if merge_result.NewZStabs is not None:
                total_new_Z_stabs += len(merge_result.NewZStabs)
            if merge_result.NewXStabs is not None:
                total_new_X_stabs += len(merge_result.NewXStabs)

        else:
            n = total_merge_map.shape[1]
            v0 = indices_to_vector(merge_sequence.IndexSequence[i][0], n)
            v1 = indices_to_vector(merge_sequence.IndexSequence[i][1], n)

            new_v0 = vector_to_indices(
                list((multiply_F2(total_merge_map, np.array([v0]).T).T)[0])
            )
            new_v1 = vector_to_indices(
                list((multiply_F2(total_merge_map, np.array([v1]).T).T)[0])
            )

            merge_result = internal_merge_by_indices(
                merge_result.Code,
                new_v0,
                new_v1,
                basis,
                merge_sequence.DepthSequence[i],
                True,
            )
            if merge_result is None:
                raise ValueError(
                    "parallel internal merges cannot be found for the given input"
                )
            total_merge_map = multiply_F2(merge_result.MergeMap, total_merge_map)
            num_new_qubits += len(merge_result.NewQubits)
            total_new_Z_stabs += len(merge_result.NewZStabs)
            total_new_X_stabs += len(merge_result.NewXStabs)

    if return_data:
        new_Z_stabilisers = range(total_new_Z_stabs)
        new_X_stabilisers = range(total_new_X_stabs)

        num_before_logicals = num_logical_qubits(C)
        num_after_logicals = num_logical_qubits(merge_result.Code)
        num_new_logicals = num_after_logicals - num_before_logicals + num_merges
        new_qubits = [i + num_data_qubits(C) for i in range(num_new_qubits)]
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
                    C,
                    merge_result.Code,
                    total_merge_map,
                    num_before_logicals,
                    num_after_logicals,
                    num_merges,
                    basis,
                )
            )
        else:
            old_Z_logicals = find_Z_basis(merge_result.Code)
            old_X_logicals = find_X_basis(merge_result.Code)
            old_X_logicals = find_paired_basis(old_Z_logicals, old_X_logicals)

        return MergeResult(
            merge_result.Code,
            total_merge_map,
            new_Z_stabilisers,
            new_X_stabilisers,
            new_qubits,
            new_Z_logicals,
            new_X_logicals,
            old_Z_logicals,
            old_X_logicals,
        )

    return merge_result.Code


def parallel_external_merges(
    C: CSScode,
    D: CSScode,
    merge_sequence: MergeSequence,
    basis: str = "Z",
    return_data: bool = False,
) -> CSScode | MergeResult:
    """Takes two CSS codeblocks and performs several external merges in parallel.

    Args:
        C: The first CSS codeblock.
        D: The second CSS codeblock.
        merge_sequence: The merges to be performed.
        basis: The basis of the measurements.
        return_data: Whether to return a MergeResult or a CSScode object.


    Returns:
        The merged CSScode, or the MergeResult object which contains the CSScode.
    """
    merge_seq = copy.deepcopy(merge_sequence)
    sum_code = direct_sum_codes(C, D)
    n_C = num_data_qubits(C)
    for indices in merge_seq.IndexSequence:
        indices[1] = [i + n_C for i in indices[1]]
    return parallel_internal_merges(sum_code, merge_seq, basis, return_data)
