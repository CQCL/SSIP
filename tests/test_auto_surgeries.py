import numpy as np
import pytest

from ssip.auto_surgeries import (
    MeasureSequence,
    MergeSequence,
    measure_single_logical_qubit,
    parallel_external_merges,
    parallel_single_logical_qubit_measurements,
)
from ssip.basic_functions import (
    compose_to_zero,
    direct_sum_codes,
    find_paired_basis,
    find_X_basis,
    find_Z_basis,
    flip_code,
    image_basis_calc,
    is_valid_CSScode,
    kernel_basis_calc,
    multiply_F2,
    num_data_qubits,
    num_logical_qubits,
    num_X_stabs,
    num_Z_stabs,
    vector_to_indices,
)
from ssip.code_examples import (
    QRM_code,
    Shor_code,
    Steane_code,
    Surface3_code,
    Toric_code,
)
from ssip.lifted_product import lift_connected_surface_codes
from ssip.monic_span_checker import is_irreducible_logical


def test_shor_singleq_measurement():
    v = [1, 0, 0, 1, 0, 0, 1, 0, 0]
    indices = vector_to_indices(v)

    merged_code = measure_single_logical_qubit(Shor_code, indices, "Z", 1, True)
    assert is_valid_CSScode(merged_code.Code)
    assert num_logical_qubits(merged_code.Code) == 0
    assert num_data_qubits(merged_code.Code) == 11

    assert len(merged_code.NewZLogicals) == 0
    assert len(merged_code.NewXLogicals) == 0
    assert len(merged_code.OldZLogicals) == 0
    assert len(merged_code.OldXLogicals) == 0

    assert len(merged_code.NewXStabs) == 0
    assert num_X_stabs(merged_code.Code) == num_X_stabs(Shor_code)

    assert num_Z_stabs(merged_code.Code) - len(merged_code.NewZStabs) == num_Z_stabs(
        Shor_code
    )

    u = np.array([[1] * (num_data_qubits(Shor_code))]).T
    u_after = multiply_F2(merged_code.MergeMap, u).T[0]

    for i in merged_code.NewQubits:
        assert not u_after[i]

    assert num_data_qubits(merged_code.Code) - len(
        merged_code.NewQubits
    ) == num_data_qubits(Shor_code)

    # higher depth
    for depth in range(2, 5):
        merged_code = measure_single_logical_qubit(Shor_code, indices, "Z", depth, True)
        assert is_valid_CSScode(merged_code.Code)
        assert num_logical_qubits(merged_code.Code) == 0

        assert len(merged_code.NewZLogicals) == 0
        assert len(merged_code.NewXLogicals) == 0
        assert len(merged_code.OldZLogicals) == 0
        assert len(merged_code.OldXLogicals) == 0

        assert num_X_stabs(merged_code.Code) == num_X_stabs(Shor_code) + len(
            merged_code.NewXStabs
        )

        assert num_Z_stabs(merged_code.Code) == num_Z_stabs(Shor_code) + len(
            merged_code.NewZStabs
        )

        u = np.array([[1] * (num_data_qubits(Shor_code))]).T
        u_after = multiply_F2(merged_code.MergeMap, u).T[0]

        for i in merged_code.NewQubits:
            assert not u_after[i]

        assert num_data_qubits(merged_code.Code) - len(
            merged_code.NewQubits
        ) == num_data_qubits(Shor_code)

    flipped_code = flip_code(Shor_code)
    merged_code = measure_single_logical_qubit(flipped_code, indices, "X", 1, True)
    assert is_valid_CSScode(merged_code.Code)
    assert num_logical_qubits(merged_code.Code) == 0
    assert num_data_qubits(merged_code.Code) == 11

    assert len(merged_code.NewZStabs) == 0
    assert num_Z_stabs(merged_code.Code) == num_Z_stabs(flipped_code)

    assert num_X_stabs(merged_code.Code) - len(merged_code.NewXStabs) == num_X_stabs(
        flipped_code
    )

    u = np.array([[1] * (num_data_qubits(flipped_code))]).T
    u_after = multiply_F2(merged_code.MergeMap, u).T[0]

    for i in merged_code.NewQubits:
        assert not u_after[i]

    assert num_data_qubits(merged_code.Code) - len(
        merged_code.NewQubits
    ) == num_data_qubits(flipped_code)

    merged_code = measure_single_logical_qubit(flipped_code, indices, "X", 1, False)
    assert is_valid_CSScode(merged_code)
    assert num_logical_qubits(merged_code) == 0
    assert num_data_qubits(merged_code) == 11

    with pytest.raises(ValueError, match="Must enter a valid Pauli string."):
        measure_single_logical_qubit(flipped_code, indices, "Y", 1, True)


def test_QRM_singleq_measurement():
    v = [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    indices = vector_to_indices(v)

    for depth in range(1, 5):
        merged_code = measure_single_logical_qubit(QRM_code, indices, "Z", depth, True)
        assert is_valid_CSScode(merged_code.Code)
        assert num_logical_qubits(merged_code.Code) == 0

        assert len(merged_code.NewZLogicals) == 0
        assert len(merged_code.NewXLogicals) == 0
        assert len(merged_code.OldZLogicals) == 0
        assert len(merged_code.OldXLogicals) == 0


def test_surface_singleq_measurement():
    v = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    indices = vector_to_indices(v)

    for depth in range(1, 5):
        merged_code = measure_single_logical_qubit(
            Surface3_code, indices, "Z", depth, True
        )
        assert is_valid_CSScode(merged_code.Code)
        assert num_logical_qubits(merged_code.Code) == 0

        assert len(merged_code.NewZLogicals) == 0
        assert len(merged_code.NewXLogicals) == 0
        assert len(merged_code.OldZLogicals) == 0
        assert len(merged_code.OldXLogicals) == 0


def test_colour_singleq_measurement():
    colour_logical = [1, 1, 1, 0, 0, 0, 0]
    indices = vector_to_indices(colour_logical)

    for depth in range(1, 5):
        merged_code = measure_single_logical_qubit(
            Steane_code, indices, "Z", depth, True
        )
        assert is_valid_CSScode(merged_code.Code)
        assert num_logical_qubits(merged_code.Code) == 0

        assert len(merged_code.NewZLogicals) == 0
        assert len(merged_code.NewXLogicals) == 0
        assert len(merged_code.OldZLogicals) == 0
        assert len(merged_code.OldXLogicals) == 0


def test_toric_singleq_measurement():
    toric_logical = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    indices = vector_to_indices(toric_logical)

    for depth in range(1, 5):
        merged_code = measure_single_logical_qubit(
            Toric_code, indices, "Z", depth, True
        )
        assert is_valid_CSScode(merged_code.Code)
        assert num_logical_qubits(merged_code.Code) == 1

        assert len(merged_code.NewZLogicals) == 0
        assert len(merged_code.NewXLogicals) == 0
        assert len(merged_code.OldZLogicals) == 1
        assert len(merged_code.OldXLogicals) == 1


def test_LCS_singleq_measurement():
    l = 1
    L = 3
    lifted_code = lift_connected_surface_codes(l, L)

    indices = [6, 8, 9]
    for depth in range(1, 5):
        merged_code = measure_single_logical_qubit(
            lifted_code, indices, "Z", depth, True
        )
        assert is_valid_CSScode(merged_code.Code)
        assert num_logical_qubits(merged_code.Code) == 2

        assert len(merged_code.NewZLogicals) == 0
        assert len(merged_code.NewXLogicals) == 0
        assert len(merged_code.OldZLogicals) == 2
        assert len(merged_code.OldXLogicals) == 2

    indices2 = [2, 9, 14]
    for depth in range(1, 5):
        merged_code = measure_single_logical_qubit(
            lifted_code, indices2, "Z", depth, True
        )
        assert is_valid_CSScode(merged_code.Code)
        assert num_logical_qubits(merged_code.Code) == 3

        assert len(merged_code.NewZLogicals) == 1
        assert len(merged_code.NewXLogicals) == 1
        assert len(merged_code.OldZLogicals) == 2
        assert len(merged_code.OldXLogicals) == 2

    l = 1
    L = 4
    lifted_code = lift_connected_surface_codes(l, L)

    indices = [10, 11, 15]
    for depth in range(1, 5):
        merged_code = measure_single_logical_qubit(
            lifted_code, indices, "Z", depth, True
        )
        assert is_valid_CSScode(merged_code.Code)
        assert num_logical_qubits(merged_code.Code) == 3

        assert len(merged_code.NewZLogicals) == 0
        assert len(merged_code.NewXLogicals) == 0
        assert len(merged_code.OldZLogicals) == 3
        assert len(merged_code.OldXLogicals) == 3

    indices2 = [3, 12, 19]
    for depth in range(1, 5):
        merged_code = measure_single_logical_qubit(
            lifted_code, indices2, "Z", depth, True
        )
        assert is_valid_CSScode(merged_code.Code)
        assert num_logical_qubits(merged_code.Code) == 4

        assert len(merged_code.NewZLogicals) == 1
        assert len(merged_code.NewXLogicals) == 1
        assert len(merged_code.OldZLogicals) == 3
        assert len(merged_code.OldXLogicals) == 3


def test_single_logical_qubit_measurements():
    toric_logical = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    indices = vector_to_indices(toric_logical)

    index_list = [indices]
    rs = [1]
    measure_sequence = MeasureSequence(index_list, rs)

    measured_code = parallel_single_logical_qubit_measurements(
        Toric_code, measure_sequence, "Z", True
    )
    assert num_logical_qubits(measured_code.Code) == 1
    assert len(measured_code.NewZStabs) == 3
    assert len(measured_code.NewXStabs) == 0
    assert len(measured_code.NewQubits) == 3

    toric_logical2 = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1]
    indices2 = vector_to_indices(toric_logical2)
    assert compose_to_zero(Toric_code.PX, np.array([toric_logical2]).T)

    index_list = [indices, indices2]
    rs = [1, 1]
    measure_sequence = MeasureSequence(index_list, rs)
    measured_code = parallel_single_logical_qubit_measurements(
        Toric_code, measure_sequence, "Z", True
    )
    assert num_logical_qubits(measured_code.Code) - len(measured_code.NewZLogicals) == 0
    assert len(measured_code.NewZStabs) == 8
    assert len(measured_code.NewXStabs) == 0
    assert len(measured_code.NewQubits) == 8

    measured_code = parallel_single_logical_qubit_measurements(
        Toric_code, measure_sequence, "Z", False
    )
    assert num_logical_qubits(measured_code) == 2


def test_parallel_merges():
    for l in range(1, 3):
        L = l + 2
        lcs_code = lift_connected_surface_codes(l, L)
        hom_basis = find_Z_basis(lcs_code)

        index_sequence = []
        for v in hom_basis:
            ind = vector_to_indices(v)
            if is_irreducible_logical(ind, lcs_code.PX):
                index_sequence.append([ind, ind])

        depths = [1] * len(index_sequence)
        merge_sequence = MergeSequence(index_sequence, depths)
        merge_data = parallel_external_merges(
            lcs_code, lcs_code, merge_sequence, "Z", True
        )

        assert is_valid_CSScode(merge_data.Code)

        before_code = direct_sum_codes(lcs_code, lcs_code)
        ker_before = np.array(kernel_basis_calc(before_code.PX)).T
        ker_after = multiply_F2(merge_data.MergeMap, ker_before)
        assert compose_to_zero(merge_data.Code.PX, ker_after)

        im_before = np.array(image_basis_calc(before_code.PZ.T)).T
        im_after = multiply_F2(merge_data.MergeMap, im_before)
        im_quotient = np.array(
            kernel_basis_calc(np.array(image_basis_calc(merge_data.Code.PZ.T)))
        )
        assert compose_to_zero(im_quotient, im_after)

    indices = [0, 1, 2]
    index_sequence = [[indices, indices]]
    depths = [1]
    merge_sequence = MergeSequence(index_sequence, depths)
    merge_data = parallel_external_merges(
        Toric_code, Toric_code, merge_sequence, "Z", True
    )
    assert num_logical_qubits(merge_data.Code) == 3

    merged_code = parallel_external_merges(
        Toric_code, Toric_code, merge_sequence, "Z", False
    )
    assert num_logical_qubits(merged_code) == 3

    flipped_code = flip_code(Toric_code)
    merge_data = parallel_external_merges(
        flipped_code, flipped_code, merge_sequence, "X", True
    )
    assert num_logical_qubits(merge_data.Code) == 3

    l = 2
    L = 4
    lcs_code = lift_connected_surface_codes(l, L)
    hom_basis = find_Z_basis(lcs_code)
    cohom_basis = find_paired_basis(hom_basis, find_X_basis(lcs_code))

    index_sequence = []
    for v in cohom_basis:
        ind = vector_to_indices(v)
        if is_irreducible_logical(ind, lcs_code.PZ):
            index_sequence.append([ind, ind])

    depths = [1] * len(index_sequence)
    merge_sequence = MergeSequence(index_sequence, depths)
    merge_data = parallel_external_merges(lcs_code, lcs_code, merge_sequence, "X", True)

    assert is_valid_CSScode(merge_data.Code)

    before_code = direct_sum_codes(lcs_code, lcs_code)
    ker_before = np.array(kernel_basis_calc(before_code.PZ)).T
    ker_after = multiply_F2(merge_data.MergeMap, ker_before)
    assert compose_to_zero(merge_data.Code.PZ, ker_after)

    im_before = np.array(image_basis_calc(before_code.PX.T)).T
    im_after = multiply_F2(merge_data.MergeMap, im_before)
    im_quotient = np.array(
        kernel_basis_calc(np.array(image_basis_calc(merge_data.Code.PX.T)))
    )
    assert compose_to_zero(im_quotient, im_after)
