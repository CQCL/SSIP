import numpy as np
import pytest

from ssip.basic_functions import (
    CSScode,
    compose_to_zero,
    direct_sum_matrices,
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
    GTri_code,
    QRM_code,
    Shor_A,
    Shor_B,
    Shor_code,
    Steane_code,
    Surface2_code,
    Surface3_code,
    Toric_A,
    Toric_code,
)
from ssip.distance import (
    Z_distance,
    code_distance,
    subsystem_distance_GAP,
)
from ssip.lifted_product import lift_connected_surface_codes
from ssip.pushouts import (
    external_merge_by_indices,
    pushout,
    pushout_by_indices,
    sandwich_middle_code,
)

# Testing the code distances of the quotiented/merged codes can take some time
TEST_PUSHOUT_CODE_DISTANCES = False

##### Tests #####


def test_sandwich_middle_code():
    # classical [3, 1, 3] repetition code
    C = np.array([[1, 1, 0], [0, 1, 1]])
    tensor_code = sandwich_middle_code(C)
    assert is_valid_CSScode(tensor_code)
    assert num_data_qubits(tensor_code) == C.shape[0] + 2 * C.shape[1]
    assert num_logical_qubits(tensor_code) == 1
    assert code_distance(tensor_code) == 2

    # classical [7,4,3] Hamming code
    C = np.array([[1, 1, 0, 1, 1, 0, 0], [1, 0, 1, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0, 1]])
    tensor_code = sandwich_middle_code(C)
    assert is_valid_CSScode(tensor_code)
    assert num_data_qubits(tensor_code) == C.shape[0] + 2 * C.shape[1]
    assert num_logical_qubits(tensor_code) == 4
    assert code_distance(tensor_code) == 2

    # classical [3,2,1] trivial code
    C = np.array([[1, 1, 0]])
    tensor_code = sandwich_middle_code(C)
    assert is_valid_CSScode(tensor_code)
    assert num_data_qubits(tensor_code) == C.shape[0] + 2 * C.shape[1]
    assert num_logical_qubits(tensor_code) == 2
    assert code_distance(tensor_code) == 1


def test_pushout_failures():
    ind = [0, 3, 6]
    with pytest.raises(ValueError, match="Must enter a valid Pauli string."):
        pushout_by_indices(Shor_code, Shor_code, ind, ind, "Y")

    with pytest.raises(ValueError, match="Must enter a valid Pauli string."):
        external_merge_by_indices(Shor_code, Shor_code, ind, ind, "Y")

    ind2 = [0, 3]
    pushout_result = pushout_by_indices(Shor_code, Shor_code, ind, ind2, "Z")
    assert pushout_result is None

    with pytest.raises(ValueError, match="Must enter a pushout depth greater than 0."):
        external_merge_by_indices(Shor_code, Shor_code, ind, ind, "Z", 0, False)


# Testing single pushouts, i.e. a quotient on the codes
def test_shor_quotient():
    qubit_map = {0: 0, 3: 3, 6: 6}
    syndrome_map = {0: 0, 1: 1}
    quotient_code = pushout(
        Shor_A, Shor_B, Shor_A, Shor_B, qubit_map, syndrome_map, True
    )
    assert is_valid_CSScode(quotient_code.Code)
    assert num_logical_qubits(quotient_code.Code) == 1
    ker_before = np.array(kernel_basis_calc(direct_sum_matrices(Shor_B, Shor_B))).T
    ker_after = multiply_F2(quotient_code.MergeMap, ker_before)
    assert compose_to_zero(quotient_code.Code.PX, ker_after)

    im_before = np.array(image_basis_calc(direct_sum_matrices(Shor_A, Shor_A))).T
    im_after = multiply_F2(quotient_code.MergeMap, im_before)
    im_quotient = np.array(
        kernel_basis_calc(np.array(image_basis_calc(quotient_code.Code.PZ.T)))
    )
    assert compose_to_zero(im_quotient, im_after)

    if TEST_PUSHOUT_CODE_DISTANCES:
        assert code_distance(quotient_code.Code) == 3

    v = [1, 0, 0, 1, 0, 0, 1, 0, 0]
    indices = vector_to_indices(v)
    quotient_code = pushout_by_indices(
        Shor_code, Shor_code, indices, indices, "Z", True
    )
    assert quotient_code is not None
    assert is_valid_CSScode(quotient_code.Code)
    assert num_logical_qubits(quotient_code.Code) == 1
    ker_after = multiply_F2(quotient_code.MergeMap, ker_before)
    assert compose_to_zero(quotient_code.Code.PX, ker_after)

    im_after = multiply_F2(quotient_code.MergeMap, im_before)
    im_quotient = np.array(
        kernel_basis_calc(np.array(image_basis_calc(quotient_code.Code.PZ.T)))
    )
    assert compose_to_zero(im_quotient, im_after)

    if TEST_PUSHOUT_CODE_DISTANCES:
        assert Z_distance(quotient_code.Code) >= 3

    new_code = CSScode(Shor_code.PX, Shor_code.PZ)
    quotient_code2 = pushout_by_indices(new_code, new_code, indices, indices, "X", True)
    assert quotient_code2 is not None
    assert is_valid_CSScode(quotient_code2.Code)
    assert np.array_equal(quotient_code.Code.PZ, quotient_code2.Code.PX)
    assert np.array_equal(quotient_code.Code.PX, quotient_code2.Code.PZ)
    ker_before = np.array(
        kernel_basis_calc(direct_sum_matrices(new_code.PZ, new_code.PZ))
    ).T
    ker_after = multiply_F2(quotient_code2.MergeMap, ker_before)
    assert compose_to_zero(quotient_code2.Code.PZ, ker_after)

    im_before = np.array(
        image_basis_calc(direct_sum_matrices(new_code.PX.T, new_code.PX.T))
    ).T
    im_after = multiply_F2(quotient_code2.MergeMap, im_before)
    im_quotient = np.array(
        kernel_basis_calc(np.array(image_basis_calc(quotient_code2.Code.PX.T)))
    )
    assert compose_to_zero(im_quotient, im_after)


def test_shor_surface_quotient():
    v1 = [1, 0, 0, 1, 0, 0, 1, 0, 0]
    ind1 = vector_to_indices(v1)
    v2 = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ind2 = vector_to_indices(v2)

    quotient_code = pushout_by_indices(Shor_code, Surface3_code, ind1, ind2, "Z", True)
    assert quotient_code is not None
    assert is_valid_CSScode(quotient_code.Code)
    assert num_logical_qubits(quotient_code.Code) == 1
    ker_before = np.array(
        kernel_basis_calc(direct_sum_matrices(Shor_code.PX, Surface3_code.PX))
    ).T
    ker_after = multiply_F2(quotient_code.MergeMap, ker_before)
    assert compose_to_zero(quotient_code.Code.PX, ker_after)

    im_before = np.array(
        image_basis_calc(direct_sum_matrices(Shor_A, Surface3_code.PZ.T))
    ).T
    im_after = multiply_F2(quotient_code.MergeMap, im_before)
    im_quotient = np.array(
        kernel_basis_calc(np.array(image_basis_calc(quotient_code.Code.PZ.T)))
    )
    assert compose_to_zero(im_quotient, im_after)

    if TEST_PUSHOUT_CODE_DISTANCES:
        assert code_distance(quotient_code.Code) >= 3

    new_code = CSScode(Shor_code.PX, Shor_code.PZ)
    new_code2 = CSScode(Surface3_code.PX, Surface3_code.PZ)
    quotient_code2 = pushout_by_indices(new_code, new_code2, ind1, ind2, "X")
    assert np.array_equal(quotient_code.Code.PZ, quotient_code2.PX)
    assert np.array_equal(quotient_code.Code.PX, quotient_code2.PZ)


def test_surface_qrm_quotient():
    logical_to_merge1 = [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    indices1 = vector_to_indices(logical_to_merge1)
    logical_to_merge2 = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    indices2 = vector_to_indices(logical_to_merge2)

    quotient_code = pushout_by_indices(
        QRM_code, Surface3_code, indices1, indices2, "Z", True
    )
    assert quotient_code is not None
    assert is_valid_CSScode(quotient_code.Code)
    assert num_logical_qubits(quotient_code.Code) == 1
    if TEST_PUSHOUT_CODE_DISTANCES:
        assert code_distance(quotient_code.Code) >= 3
    ker_before = np.array(
        kernel_basis_calc(direct_sum_matrices(QRM_code.PX, Surface3_code.PX))
    ).T
    ker_after = multiply_F2(quotient_code.MergeMap, ker_before)
    assert compose_to_zero(quotient_code.Code.PX, ker_after)

    im_before = np.array(
        image_basis_calc(direct_sum_matrices(QRM_code.PZ.T, Surface3_code.PZ.T))
    ).T
    im_after = multiply_F2(quotient_code.MergeMap, im_before)
    im_quotient = np.array(
        kernel_basis_calc(np.array(image_basis_calc(quotient_code.Code.PZ.T)))
    )
    assert compose_to_zero(im_quotient, im_after)


def test_shor_qrm_quotient():
    qrm_logical = [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ind1 = vector_to_indices(qrm_logical)
    shor_logical = [1, 0, 0, 1, 0, 0, 1, 0, 0]
    ind2 = vector_to_indices(shor_logical)

    quotient_code = pushout_by_indices(QRM_code, Shor_code, ind1, ind2, "Z", True)
    assert quotient_code is not None
    assert is_valid_CSScode(quotient_code.Code)
    assert num_logical_qubits(quotient_code.Code) == 1
    if TEST_PUSHOUT_CODE_DISTANCES:
        assert code_distance(quotient_code.Code) >= 3
    ker_before = np.array(
        kernel_basis_calc(direct_sum_matrices(QRM_code.PX, Shor_code.PX))
    ).T
    ker_after = multiply_F2(quotient_code.MergeMap, ker_before)
    assert compose_to_zero(quotient_code.Code.PX, ker_after)

    im_before = np.array(
        image_basis_calc(direct_sum_matrices(QRM_code.PZ.T, Shor_code.PZ.T))
    ).T
    im_after = multiply_F2(quotient_code.MergeMap, im_before)
    im_quotient = np.array(
        kernel_basis_calc(np.array(image_basis_calc(quotient_code.Code.PZ.T)))
    )
    assert compose_to_zero(im_quotient, im_after)


def test_colour_surface_quotient():
    colour_logical = [1, 1, 1, 0, 0, 0, 0]
    ind1 = vector_to_indices(colour_logical)
    surface_logical = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ind2 = vector_to_indices(surface_logical)
    quotient_code = pushout_by_indices(
        Steane_code, Surface3_code, ind1, ind2, "Z", True
    )
    assert quotient_code is not None
    assert is_valid_CSScode(quotient_code.Code)
    assert num_logical_qubits(quotient_code.Code) == 1
    if TEST_PUSHOUT_CODE_DISTANCES:
        assert code_distance(quotient_code.Code) >= 3
    ker_before = np.array(
        kernel_basis_calc(direct_sum_matrices(Steane_code.PX, Surface3_code.PX))
    ).T
    ker_after = multiply_F2(quotient_code.MergeMap, ker_before)
    assert compose_to_zero(quotient_code.Code.PX, ker_after)

    im_before = np.array(
        image_basis_calc(direct_sum_matrices(Steane_code.PZ.T, Surface3_code.PZ.T))
    ).T
    im_after = multiply_F2(quotient_code.MergeMap, im_before)
    im_quotient = np.array(
        kernel_basis_calc(np.array(image_basis_calc(quotient_code.Code.PZ.T)))
    )
    assert compose_to_zero(im_quotient, im_after)


def test_colour_shor_quotient():
    colour_logical = [1, 1, 1, 0, 0, 0, 0]
    ind1 = vector_to_indices(colour_logical)
    shor_logical = [1, 0, 0, 1, 0, 0, 1, 0, 0]
    ind2 = vector_to_indices(shor_logical)
    quotient_code = pushout_by_indices(Steane_code, Shor_code, ind1, ind2, "Z", True)
    assert quotient_code is not None
    assert is_valid_CSScode(quotient_code.Code)
    assert num_logical_qubits(quotient_code.Code) == 1
    if TEST_PUSHOUT_CODE_DISTANCES:
        assert code_distance(quotient_code.Code) >= 3
    ker_before = np.array(
        kernel_basis_calc(direct_sum_matrices(Steane_code.PX, Shor_code.PX))
    ).T
    ker_after = multiply_F2(quotient_code.MergeMap, ker_before)
    assert compose_to_zero(quotient_code.Code.PX, ker_after)

    im_before = np.array(
        image_basis_calc(direct_sum_matrices(Steane_code.PZ.T, Shor_code.PZ.T))
    ).T
    im_after = multiply_F2(quotient_code.MergeMap, im_before)
    im_quotient = np.array(
        kernel_basis_calc(np.array(image_basis_calc(quotient_code.Code.PZ.T)))
    )
    assert compose_to_zero(im_quotient, im_after)


def test_gtri_surface_quotient():
    surface_logical = [1, 1, 0, 0, 0]
    ind1 = vector_to_indices(surface_logical)

    gtri_logical = [1, 1, 0, 0, 0, 0, 0, 0]
    ind2 = vector_to_indices(gtri_logical)
    quotient_code = pushout_by_indices(Surface2_code, GTri_code, ind1, ind2, "Z", True)
    assert quotient_code is not None
    assert is_valid_CSScode(quotient_code.Code)
    assert num_logical_qubits(quotient_code.Code) == 3
    if TEST_PUSHOUT_CODE_DISTANCES:
        assert code_distance(quotient_code.Code) >= 2
    ker_before = np.array(
        kernel_basis_calc(direct_sum_matrices(Surface2_code.PX, GTri_code.PX))
    ).T
    ker_after = multiply_F2(quotient_code.MergeMap, ker_before)
    assert compose_to_zero(quotient_code.Code.PX, ker_after)

    im_before = np.array(
        image_basis_calc(direct_sum_matrices(Surface2_code.PZ.T, GTri_code.PZ.T))
    ).T
    im_after = multiply_F2(quotient_code.MergeMap, im_before)
    im_quotient = np.array(
        kernel_basis_calc(np.array(image_basis_calc(quotient_code.Code.PZ.T)))
    )
    assert compose_to_zero(im_quotient, im_after)


# Testing two pushouts using the tensor product, i.e. a code merge
def test_shor_merge():
    v = [1, 0, 0, 1, 0, 0, 1, 0, 0]
    indices = vector_to_indices(v)
    merged_code = external_merge_by_indices(
        Shor_code, Shor_code, indices, indices, "Z", 1, True
    )
    assert is_valid_CSScode(merged_code.Code)
    assert num_logical_qubits(merged_code.Code) == 1
    ker_before = np.array(
        kernel_basis_calc(direct_sum_matrices(Shor_code.PX, Shor_code.PX))
    ).T
    ker_after = multiply_F2(merged_code.MergeMap, ker_before)
    assert compose_to_zero(merged_code.Code.PX, ker_after)

    im_before = np.array(
        image_basis_calc(direct_sum_matrices(Shor_code.PZ.T, Shor_code.PZ.T))
    ).T
    im_after = multiply_F2(merged_code.MergeMap, im_before)
    im_quotient = np.array(
        kernel_basis_calc(np.array(image_basis_calc(merged_code.Code.PZ.T)))
    )
    assert compose_to_zero(im_quotient, im_after)

    assert num_X_stabs(merged_code.Code) == 2 * num_X_stabs(Shor_code)

    assert num_Z_stabs(merged_code.Code) - len(
        merged_code.NewZStabs
    ) == 2 * num_Z_stabs(Shor_code)

    u = np.array([[1] * (2 * num_data_qubits(Shor_code))]).T
    u_after = multiply_F2(merged_code.MergeMap, u).T[0]

    for i in merged_code.NewQubits:
        assert not u_after[i]

    assert num_data_qubits(merged_code.Code) - len(
        merged_code.NewQubits
    ) == 2 * num_data_qubits(Shor_code)

    # higher merge depth
    for depth in range(2, 5):
        merged_code = external_merge_by_indices(
            Shor_code, Shor_code, indices, indices, "Z", depth, True
        )
        assert is_valid_CSScode(merged_code.Code)
        assert num_logical_qubits(merged_code.Code) == 1
        ker_before = np.array(
            kernel_basis_calc(direct_sum_matrices(Shor_code.PX, Shor_code.PX))
        ).T
        ker_after = multiply_F2(merged_code.MergeMap, ker_before)
        assert compose_to_zero(merged_code.Code.PX, ker_after)

        im_before = np.array(
            image_basis_calc(direct_sum_matrices(Shor_code.PZ.T, Shor_code.PZ.T))
        ).T
        im_after = multiply_F2(merged_code.MergeMap, im_before)
        im_quotient = np.array(
            kernel_basis_calc(np.array(image_basis_calc(merged_code.Code.PZ.T)))
        )
        assert compose_to_zero(im_quotient, im_after)

        assert num_X_stabs(merged_code.Code) - len(
            merged_code.NewXStabs
        ) == 2 * num_X_stabs(Shor_code)

        assert num_Z_stabs(merged_code.Code) - len(
            merged_code.NewZStabs
        ) == 2 * num_Z_stabs(Shor_code)

        u_after = multiply_F2(merged_code.MergeMap, u).T[0]

        for i in merged_code.NewQubits:
            assert not u_after[i]

        assert num_data_qubits(merged_code.Code) - len(
            merged_code.NewQubits
        ) == 2 * num_data_qubits(Shor_code)


def test_shor_surface_merge():
    v1 = [1, 0, 0, 1, 0, 0, 1, 0, 0]
    ind1 = vector_to_indices(v1)
    v2 = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ind2 = vector_to_indices(v2)

    # without MergeResult...
    merged_code = external_merge_by_indices(Shor_code, Surface3_code, ind1, ind2)
    assert merged_code is not None
    assert is_valid_CSScode(merged_code)
    assert num_logical_qubits(merged_code) == 1

    # ...with MergeResult
    merged_code = external_merge_by_indices(
        Shor_code, Surface3_code, ind1, ind2, "Z", 1, True
    )
    assert merged_code is not None
    assert is_valid_CSScode(merged_code.Code)
    assert num_logical_qubits(merged_code.Code) == 1
    if TEST_PUSHOUT_CODE_DISTANCES:
        assert code_distance(merged_code.Code) >= 3

    ker_before = np.array(
        kernel_basis_calc(direct_sum_matrices(Shor_code.PX, Surface3_code.PX))
    ).T
    ker_after = multiply_F2(merged_code.MergeMap, ker_before)
    assert compose_to_zero(merged_code.Code.PX, ker_after)

    im_before = np.array(
        image_basis_calc(direct_sum_matrices(Shor_code.PZ.T, Surface3_code.PZ.T))
    ).T
    im_after = multiply_F2(merged_code.MergeMap, im_before)
    im_quotient = np.array(
        kernel_basis_calc(np.array(image_basis_calc(merged_code.Code.PZ.T)))
    )
    assert compose_to_zero(im_quotient, im_after)

    assert num_X_stabs(merged_code.Code) - len(merged_code.NewXStabs) == num_X_stabs(
        Shor_code
    ) + num_X_stabs(Surface3_code)

    assert num_Z_stabs(merged_code.Code) - len(merged_code.NewZStabs) == num_Z_stabs(
        Shor_code
    ) + num_Z_stabs(Surface3_code)

    # higher merge depth
    for depth in range(2, 5):
        merged_code = external_merge_by_indices(
            Shor_code, Surface3_code, ind1, ind2, "Z", depth, True
        )
        assert is_valid_CSScode(merged_code.Code)
        assert num_logical_qubits(merged_code.Code) == 1
        if TEST_PUSHOUT_CODE_DISTANCES:
            assert code_distance(merged_code.Code) >= 3
        ker_before = np.array(
            kernel_basis_calc(direct_sum_matrices(Shor_code.PX, Surface3_code.PX))
        ).T
        ker_after = multiply_F2(merged_code.MergeMap, ker_before)
        assert compose_to_zero(merged_code.Code.PX, ker_after)

        im_before = np.array(
            image_basis_calc(direct_sum_matrices(Shor_code.PZ.T, Surface3_code.PZ.T))
        ).T
        im_after = multiply_F2(merged_code.MergeMap, im_before)
        im_quotient = np.array(
            kernel_basis_calc(np.array(image_basis_calc(merged_code.Code.PZ.T)))
        )
        assert compose_to_zero(im_quotient, im_after)

        assert num_X_stabs(merged_code.Code) - len(
            merged_code.NewXStabs
        ) == num_X_stabs(Shor_code) + num_X_stabs(Surface3_code)

        assert num_Z_stabs(merged_code.Code) - len(
            merged_code.NewZStabs
        ) == num_Z_stabs(Shor_code) + num_Z_stabs(Surface3_code)


def test_flipped_codes_merge():
    v1 = [1, 0, 0, 1, 0, 0, 1, 0, 0]
    ind1 = vector_to_indices(v1)
    v2 = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ind2 = vector_to_indices(v2)

    merged_code = external_merge_by_indices(
        flip_code(Shor_code), flip_code(Surface3_code), ind1, ind2, "X"
    )
    assert merged_code is not None
    assert is_valid_CSScode(merged_code)
    assert num_logical_qubits(merged_code) == 1

    ind3 = [0, 1, 3]
    fail_merge = external_merge_by_indices(
        flip_code(Shor_code), flip_code(Surface3_code), ind1, ind3, "X"
    )
    assert fail_merge is None

    merged_code = external_merge_by_indices(
        flip_code(Shor_code), flip_code(Surface3_code), ind1, ind2, "X", 1, True
    )
    assert merged_code.Code is not None
    assert is_valid_CSScode(merged_code.Code)
    assert num_logical_qubits(merged_code.Code) == 1


def test_surface_qrm_merge():
    logical_to_merge1 = [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    indices1 = vector_to_indices(logical_to_merge1)
    logical_to_merge2 = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    indices2 = vector_to_indices(logical_to_merge2)

    merged_code = external_merge_by_indices(
        QRM_code, Surface3_code, indices1, indices2, "Z", 1, True
    )
    assert merged_code is not None
    assert is_valid_CSScode(merged_code.Code)
    assert num_logical_qubits(merged_code.Code) == 1
    if TEST_PUSHOUT_CODE_DISTANCES:
        assert code_distance(merged_code.Code) >= 3

    ker_before = np.array(
        kernel_basis_calc(direct_sum_matrices(QRM_code.PX, Surface3_code.PX))
    ).T
    ker_after = multiply_F2(merged_code.MergeMap, ker_before)
    assert compose_to_zero(merged_code.Code.PX, ker_after)

    im_before = np.array(
        image_basis_calc(direct_sum_matrices(QRM_code.PZ.T, Surface3_code.PZ.T))
    ).T
    im_after = multiply_F2(merged_code.MergeMap, im_before)
    im_quotient = np.array(
        kernel_basis_calc(np.array(image_basis_calc(merged_code.Code.PZ.T)))
    )
    assert compose_to_zero(im_quotient, im_after)

    # higher merge depth
    for depth in range(2, 5):
        merged_code = external_merge_by_indices(
            QRM_code, Surface3_code, indices1, indices2, "Z", depth, True
        )
        assert is_valid_CSScode(merged_code.Code)
        assert num_logical_qubits(merged_code.Code) == 1
        if TEST_PUSHOUT_CODE_DISTANCES:
            assert code_distance(merged_code.Code) >= 3
        ker_before = np.array(
            kernel_basis_calc(direct_sum_matrices(QRM_code.PX, Surface3_code.PX))
        ).T
        ker_after = multiply_F2(merged_code.MergeMap, ker_before)
        assert compose_to_zero(merged_code.Code.PX, ker_after)

        im_before = np.array(
            image_basis_calc(direct_sum_matrices(QRM_code.PZ.T, Surface3_code.PZ.T))
        ).T
        im_after = multiply_F2(merged_code.MergeMap, im_before)
        im_quotient = np.array(
            kernel_basis_calc(np.array(image_basis_calc(merged_code.Code.PZ.T)))
        )
        assert compose_to_zero(im_quotient, im_after)


def test_colour_surface_merge():
    colour_logical = [1, 1, 1, 0, 0, 0, 0]
    ind1 = vector_to_indices(colour_logical)
    surface_logical = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ind2 = vector_to_indices(surface_logical)
    merged_code = external_merge_by_indices(
        Steane_code, Surface3_code, ind1, ind2, "Z", 1, True
    )
    assert merged_code is not None
    assert is_valid_CSScode(merged_code.Code)
    assert num_logical_qubits(merged_code.Code) == 1
    if TEST_PUSHOUT_CODE_DISTANCES:
        assert code_distance(merged_code.Code) >= 3

    ker_before = np.array(
        kernel_basis_calc(direct_sum_matrices(Steane_code.PX, Surface3_code.PX))
    ).T
    ker_after = multiply_F2(merged_code.MergeMap, ker_before)
    assert compose_to_zero(merged_code.Code.PX, ker_after)

    im_before = np.array(
        image_basis_calc(direct_sum_matrices(Steane_code.PZ.T, Surface3_code.PZ.T))
    ).T
    im_after = multiply_F2(merged_code.MergeMap, im_before)
    im_quotient = np.array(
        kernel_basis_calc(np.array(image_basis_calc(merged_code.Code.PZ.T)))
    )
    assert compose_to_zero(im_quotient, im_after)

    # higher merge depth
    for depth in range(2, 5):
        merged_code = external_merge_by_indices(
            Steane_code, Surface3_code, ind1, ind2, "Z", depth, True
        )
        assert is_valid_CSScode(merged_code.Code)
        assert num_logical_qubits(merged_code.Code) == 1
        if TEST_PUSHOUT_CODE_DISTANCES:
            assert code_distance(merged_code.Code) >= 3
        ker_before = np.array(
            kernel_basis_calc(direct_sum_matrices(Steane_code.PX, Surface3_code.PX))
        ).T
        ker_after = multiply_F2(merged_code.MergeMap, ker_before)
        assert compose_to_zero(merged_code.Code.PX, ker_after)

        im_before = np.array(
            image_basis_calc(direct_sum_matrices(Steane_code.PZ.T, Surface3_code.PZ.T))
        ).T
        im_after = multiply_F2(merged_code.MergeMap, im_before)
        im_quotient = np.array(
            kernel_basis_calc(np.array(image_basis_calc(merged_code.Code.PZ.T)))
        )
        assert compose_to_zero(im_quotient, im_after)


def test_colour_shor_merge():
    colour_logical = [1, 1, 1, 0, 0, 0, 0]
    ind1 = vector_to_indices(colour_logical)
    shor_logical = [1, 0, 0, 1, 0, 0, 1, 0, 0]
    ind2 = vector_to_indices(shor_logical)
    merged_code = external_merge_by_indices(
        Steane_code, Shor_code, ind1, ind2, "Z", 1, True
    )
    assert merged_code is not None
    assert is_valid_CSScode(merged_code.Code)
    assert num_logical_qubits(merged_code.Code) == 1
    if TEST_PUSHOUT_CODE_DISTANCES:
        assert code_distance(merged_code.Code) >= 3

    ker_before = np.array(
        kernel_basis_calc(direct_sum_matrices(Steane_code.PX, Shor_code.PX))
    ).T
    ker_after = multiply_F2(merged_code.MergeMap, ker_before)
    assert compose_to_zero(merged_code.Code.PX, ker_after)

    im_before = np.array(
        image_basis_calc(direct_sum_matrices(Steane_code.PZ.T, Shor_code.PZ.T))
    ).T
    im_after = multiply_F2(merged_code.MergeMap, im_before)
    im_quotient = np.array(
        kernel_basis_calc(np.array(image_basis_calc(merged_code.Code.PZ.T)))
    )
    assert compose_to_zero(im_quotient, im_after)

    # higher merge depth
    for depth in range(2, 5):
        merged_code = external_merge_by_indices(
            Steane_code, Shor_code, ind1, ind2, "Z", depth, True
        )
        assert is_valid_CSScode(merged_code.Code)
        assert num_logical_qubits(merged_code.Code) == 1
        if TEST_PUSHOUT_CODE_DISTANCES:
            assert code_distance(merged_code.Code) >= 3
        ker_before = np.array(
            kernel_basis_calc(direct_sum_matrices(Steane_code.PX, Shor_code.PX))
        ).T
        ker_after = multiply_F2(merged_code.MergeMap, ker_before)
        assert compose_to_zero(merged_code.Code.PX, ker_after)

        im_before = np.array(
            image_basis_calc(direct_sum_matrices(Steane_code.PZ.T, Shor_code.PZ.T))
        ).T
        im_after = multiply_F2(merged_code.MergeMap, im_before)
        im_quotient = np.array(
            kernel_basis_calc(np.array(image_basis_calc(merged_code.Code.PZ.T)))
        )
        assert compose_to_zero(im_quotient, im_after)


def test_gtri_surface_merge():
    surface_logical = [1, 1, 0, 0, 0]
    ind1 = vector_to_indices(surface_logical)

    gtri_logical = [1, 1, 0, 0, 0, 0, 0, 0]
    ind2 = vector_to_indices(gtri_logical)

    merged_code = external_merge_by_indices(
        Surface2_code, GTri_code, ind1, ind2, "Z", 1, True
    )
    assert merged_code is not None
    assert is_valid_CSScode(merged_code.Code)
    assert num_logical_qubits(merged_code.Code) == 3
    if TEST_PUSHOUT_CODE_DISTANCES:
        assert code_distance(merged_code.Code) >= 2

    ker_before = np.array(
        kernel_basis_calc(direct_sum_matrices(Surface2_code.PX, GTri_code.PX))
    ).T
    ker_after = multiply_F2(merged_code.MergeMap, ker_before)
    assert compose_to_zero(merged_code.Code.PX, ker_after)

    im_before = np.array(
        image_basis_calc(direct_sum_matrices(Surface2_code.PZ.T, GTri_code.PZ.T))
    ).T
    im_after = multiply_F2(merged_code.MergeMap, im_before)
    im_quotient = np.array(
        kernel_basis_calc(np.array(image_basis_calc(merged_code.Code.PZ.T)))
    )
    assert compose_to_zero(im_quotient, im_after)


def test_toric_merge():
    toric_logical = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    indices = vector_to_indices(toric_logical)

    merged_code = external_merge_by_indices(
        Toric_code, Toric_code, indices, indices, "Z", 1, True
    )
    assert merged_code is not None
    assert is_valid_CSScode(merged_code.Code)
    assert num_logical_qubits(merged_code.Code) == 3
    if TEST_PUSHOUT_CODE_DISTANCES:
        assert code_distance(merged_code.Code) >= 3

    ker_before = np.array(
        kernel_basis_calc(direct_sum_matrices(Toric_code.PX, Toric_code.PX))
    ).T
    ker_after = multiply_F2(merged_code.MergeMap, ker_before)
    assert compose_to_zero(merged_code.Code.PX, ker_after)

    im_before = np.array(
        image_basis_calc(direct_sum_matrices(Toric_code.PZ.T, Toric_code.PZ.T))
    ).T
    im_after = multiply_F2(merged_code.MergeMap, im_before)
    im_quotient = np.array(
        kernel_basis_calc(np.array(image_basis_calc(merged_code.Code.PZ.T)))
    )
    assert compose_to_zero(im_quotient, im_after)


# Modified the X stabiliser generators of the toric code by adding rows,
# leaves codespace invariant but changes the merge slightly.
def test_modified_toric_merge():
    new_toric_B = np.array(
        [
            [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1],
        ]
    )
    new_toric_code = CSScode(Toric_A.T, new_toric_B)

    toric_logical = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    indices = vector_to_indices(toric_logical)

    merged_code = external_merge_by_indices(
        new_toric_code, new_toric_code, indices, indices, "Z", 1, True
    )
    assert is_valid_CSScode(merged_code.Code)
    assert num_logical_qubits(merged_code.Code) == 3
    if TEST_PUSHOUT_CODE_DISTANCES:
        assert code_distance(merged_code.Code) >= 3

    ker_before = np.array(
        kernel_basis_calc(direct_sum_matrices(new_toric_code.PX, new_toric_code.PX))
    ).T
    ker_after = multiply_F2(merged_code.MergeMap, ker_before)
    assert compose_to_zero(merged_code.Code.PX, ker_after)

    im_before = np.array(
        image_basis_calc(direct_sum_matrices(new_toric_code.PZ.T, new_toric_code.PZ.T))
    ).T
    im_after = multiply_F2(merged_code.MergeMap, im_before)
    im_quotient = np.array(
        kernel_basis_calc(np.array(image_basis_calc(merged_code.Code.PZ.T)))
    )
    assert compose_to_zero(im_quotient, im_after)


def test_LCS_code_merge():
    # [[15, 3, 3]] LCS code
    l = 1
    L = 3

    lifted_code = lift_connected_surface_codes(l, L)
    indices1 = [6, 8, 9]
    merged_code = external_merge_by_indices(
        lifted_code, lifted_code, indices1, indices1, "Z", 1, True
    )
    assert is_valid_CSScode(merged_code.Code)
    assert num_logical_qubits(merged_code.Code) == 5
    assert len(merged_code.NewZLogicals) == 0
    assert len(merged_code.NewXLogicals) == 0
    if TEST_PUSHOUT_CODE_DISTANCES:
        assert code_distance(merged_code.Code) >= 3

    ker_before = np.array(
        kernel_basis_calc(direct_sum_matrices(lifted_code.PX, lifted_code.PX))
    ).T
    ker_after = multiply_F2(merged_code.MergeMap, ker_before)
    assert compose_to_zero(merged_code.Code.PX, ker_after)

    im_before = np.array(
        image_basis_calc(direct_sum_matrices(lifted_code.PZ.T, lifted_code.PZ.T))
    ).T
    im_after = multiply_F2(merged_code.MergeMap, im_before)
    im_quotient = np.array(
        kernel_basis_calc(np.array(image_basis_calc(merged_code.Code.PZ.T)))
    )
    assert compose_to_zero(im_quotient, im_after)

    quotient_data = pushout_by_indices(
        lifted_code, lifted_code, indices1, indices1, "Z", True
    )
    assert is_valid_CSScode(quotient_data.Code)
    assert len(quotient_data.NewZLogicals) == 0
    assert len(quotient_data.NewXLogicals) == 0

    ker_after = multiply_F2(quotient_data.MergeMap, ker_before)
    assert compose_to_zero(quotient_data.Code.PX, ker_after)

    im_after = multiply_F2(quotient_data.MergeMap, im_before)
    im_quotient = np.array(
        kernel_basis_calc(np.array(image_basis_calc(quotient_data.Code.PZ.T)))
    )
    assert compose_to_zero(im_quotient, im_after)

    # weird counterexample
    indices2 = [2, 9, 14]
    merged_code = external_merge_by_indices(
        lifted_code, lifted_code, indices2, indices2, "Z", 1, True
    )
    assert is_valid_CSScode(merged_code.Code)
    assert num_logical_qubits(merged_code.Code) == 6
    if TEST_PUSHOUT_CODE_DISTANCES:
        assert code_distance(merged_code.Code) >= 2

    ker_after = multiply_F2(merged_code.MergeMap, ker_before)
    assert compose_to_zero(merged_code.Code.PX, ker_after)

    im_after = multiply_F2(merged_code.MergeMap, im_before)
    im_quotient = np.array(
        kernel_basis_calc(np.array(image_basis_calc(merged_code.Code.PZ.T)))
    )
    assert compose_to_zero(im_quotient, im_after)

    assert (
        subsystem_distance_GAP(
            merged_code.Code,
            merged_code.NewZLogicals,
            merged_code.NewXLogicals,
        )
        == 3
    )

    quotient_data = pushout_by_indices(
        lifted_code, lifted_code, indices2, indices2, "Z", True
    )
    assert is_valid_CSScode(quotient_data.Code)
    assert len(quotient_data.NewZLogicals) == 1
    assert len(quotient_data.NewXLogicals) == 1

    ker_after = multiply_F2(quotient_data.MergeMap, ker_before)
    assert compose_to_zero(quotient_data.Code.PX, ker_after)

    im_after = multiply_F2(quotient_data.MergeMap, im_before)
    im_quotient = np.array(
        kernel_basis_calc(np.array(image_basis_calc(quotient_data.Code.PZ.T)))
    )
    assert compose_to_zero(im_quotient, im_after)

    # different starting code
    l = 1
    L = 4
    lifted_code = lift_connected_surface_codes(l, L)
    indices1 = [10, 11, 15]
    merged_code = external_merge_by_indices(
        lifted_code, lifted_code, indices1, indices1, "Z", 1, True
    )
    assert is_valid_CSScode(merged_code.Code)
    assert num_logical_qubits(merged_code.Code) == 7

    ker_before = np.array(
        kernel_basis_calc(direct_sum_matrices(lifted_code.PX, lifted_code.PX))
    ).T
    ker_after = multiply_F2(merged_code.MergeMap, ker_before)
    assert compose_to_zero(merged_code.Code.PX, ker_after)

    im_before = np.array(
        image_basis_calc(direct_sum_matrices(lifted_code.PZ.T, lifted_code.PZ.T))
    ).T
    im_after = multiply_F2(merged_code.MergeMap, im_before)
    im_quotient = np.array(
        kernel_basis_calc(np.array(image_basis_calc(merged_code.Code.PZ.T)))
    )
    assert compose_to_zero(im_quotient, im_after)

    assert (
        subsystem_distance_GAP(
            merged_code.Code,
            merged_code.NewZLogicals,
            merged_code.NewXLogicals,
        )
        == 3
    )

    quotient_data = pushout_by_indices(
        lifted_code, lifted_code, indices1, indices1, "Z", True
    )
    assert is_valid_CSScode(quotient_data.Code)
    assert len(quotient_data.NewZLogicals) == 0
    assert len(quotient_data.NewXLogicals) == 0

    ker_after = multiply_F2(quotient_data.MergeMap, ker_before)
    assert compose_to_zero(quotient_data.Code.PX, ker_after)

    im_after = multiply_F2(quotient_data.MergeMap, im_before)
    im_quotient = np.array(
        kernel_basis_calc(np.array(image_basis_calc(quotient_data.Code.PZ.T)))
    )
    assert compose_to_zero(im_quotient, im_after)

    # similar weird counterexample
    indices2 = [3, 12, 19]
    merged_code = external_merge_by_indices(
        lifted_code, lifted_code, indices2, indices2, "Z", 1, True
    )
    assert is_valid_CSScode(merged_code.Code)
    assert num_logical_qubits(merged_code.Code) == 8

    ker_after = multiply_F2(merged_code.MergeMap, ker_before)
    assert compose_to_zero(merged_code.Code.PX, ker_after)

    im_after = multiply_F2(merged_code.MergeMap, im_before)
    im_quotient = np.array(
        kernel_basis_calc(np.array(image_basis_calc(merged_code.Code.PZ.T)))
    )
    assert compose_to_zero(im_quotient, im_after)

    quotient_data = pushout_by_indices(
        lifted_code, lifted_code, indices2, indices2, "Z", True
    )
    assert is_valid_CSScode(quotient_data.Code)
    assert len(quotient_data.NewZLogicals) == 1
    assert len(quotient_data.NewXLogicals) == 1

    ker_after = multiply_F2(quotient_data.MergeMap, ker_before)
    assert compose_to_zero(quotient_data.Code.PX, ker_after)

    im_after = multiply_F2(quotient_data.MergeMap, im_before)
    im_quotient = np.array(
        kernel_basis_calc(np.array(image_basis_calc(quotient_data.Code.PZ.T)))
    )
    assert compose_to_zero(im_quotient, im_after)


# Teleport a transversal T gate into a lift-connected surface code
def test_LCS_QRM_code_merge():
    l = 1
    L = 3

    lifted_code = lift_connected_surface_codes(l, L)
    indices1 = [6, 8, 9]
    indices2 = [0, 1, 2]

    merged_code = external_merge_by_indices(
        lifted_code, QRM_code, indices1, indices2, "Z", 1, True
    )
    assert is_valid_CSScode(merged_code.Code)
    assert num_logical_qubits(merged_code.Code) == 3
    assert num_data_qubits(merged_code.Code) == 32
    if TEST_PUSHOUT_CODE_DISTANCES:
        assert code_distance(merged_code.Code) >= 3

    ker_before = np.array(
        kernel_basis_calc(direct_sum_matrices(lifted_code.PX, QRM_code.PX))
    ).T
    ker_after = multiply_F2(merged_code.MergeMap, ker_before)
    assert compose_to_zero(merged_code.Code.PX, ker_after)

    im_before = np.array(
        image_basis_calc(direct_sum_matrices(lifted_code.PZ.T, QRM_code.PZ.T))
    ).T
    im_after = multiply_F2(merged_code.MergeMap, im_before)
    im_quotient = np.array(
        kernel_basis_calc(np.array(image_basis_calc(merged_code.Code.PZ.T)))
    )
    assert compose_to_zero(im_quotient, im_after)
