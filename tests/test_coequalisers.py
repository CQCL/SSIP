import numpy as np
import pytest

from ssip.basic_functions import (
    compose_to_zero,
    direct_sum_codes,
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
from ssip.code_examples import CSScode, Shor_code, Surface3_code, Twist_code
from ssip.coequalisers import (
    coequaliser,
    coequaliser_by_indices,
    internal_merge_by_indices,
)
from ssip.distance import Z_distance, code_distance
from ssip.lifted_product import lift_connected_surface_codes
from ssip.merge_result import find_logical_splitting


def test_failed_coequaliser():
    C = direct_sum_codes(Surface3_code, Surface3_code)
    indices1 = [0, 1, 2]
    indices2 = [13, 14, 15]

    with pytest.raises(ValueError, match="Must enter a valid Pauli string."):
        coequaliser_by_indices(C, indices1, indices2, basis="Y", return_data=False)

    coeq_code = coequaliser_by_indices(
        C, indices1, indices2, basis="Z", return_data=False
    )
    assert coeq_code is not None
    assert is_valid_CSScode(coeq_code)

    coeq_data = coequaliser_by_indices(
        C, indices1, indices2, basis="Z", return_data=True
    )
    assert coeq_data.Code is not None
    assert is_valid_CSScode(coeq_data.Code)

    coeq_code = coequaliser_by_indices(
        C, indices1, indices2[:-1], basis="Z", return_data=False
    )
    assert coeq_code is None

    flipC = flip_code(C)
    coeq_data = coequaliser_by_indices(
        flipC, indices1, indices2, basis="X", return_data=True
    )
    assert coeq_data.Code is not None
    assert is_valid_CSScode(coeq_data.Code)

    with pytest.raises(ValueError, match="Must enter a valid Pauli string."):
        internal_merge_by_indices(C, indices1, indices2, basis="Y")

    merged_code = internal_merge_by_indices(
        C, indices1, indices2[:-1], basis="Z", return_data=False
    )
    assert merged_code is None

    merged_code = internal_merge_by_indices(
        C, indices1, indices1, basis="Z", return_data=False
    )
    assert merged_code is None

    C2 = direct_sum_codes(C, C)
    ind1 = [0, 1, 2, 13, 14, 15]
    ind2 = [26, 27, 28, 39, 40, 41]
    with pytest.raises(
        ValueError, match="Internal merge error, logical operators are not irreducible"
    ):
        internal_merge_by_indices(C2, ind1, ind2, return_data=True)


# miraculously, we can perform a quotient on the twisted toric code which still yields
# a code with distance 3.
def test_twisted_toric_coequaliser():
    v1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    v2 = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    indices1 = vector_to_indices(v1)
    indices2 = vector_to_indices(v2)

    quotient_code = coequaliser_by_indices(Twist_code, indices1, indices2)
    assert quotient_code is not None
    assert is_valid_CSScode(quotient_code)
    assert num_logical_qubits(quotient_code) == 1
    assert code_distance(quotient_code) == 3

    temp_code = CSScode(Twist_code.PX, Twist_code.PZ)
    quotient_code2 = coequaliser_by_indices(temp_code, indices1, indices2, "X")
    assert quotient_code2 is not None
    assert is_valid_CSScode(quotient_code2)
    assert np.array_equal(quotient_code.PZ, quotient_code2.PX)
    assert np.array_equal(quotient_code.PX, quotient_code2.PZ)

    v2 = [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0]
    indices2 = vector_to_indices(v2)

    quotient_code = coequaliser_by_indices(Twist_code, indices1, indices2)
    assert quotient_code is not None
    assert is_valid_CSScode(quotient_code)
    assert num_logical_qubits(quotient_code) == 1
    assert code_distance(quotient_code) == 3

    # this example yields an invalid merged code, with only distance 1,
    # as there are duplicate X-syndromes. Currently returns None to indicate
    # catastrophic failure in merging.
    merged_code = internal_merge_by_indices(Twist_code, indices1, indices2)
    assert merged_code is None


# A pathological counterexample to the conjecture that coequalisers using irreducible
# logical operator subcomplexes always yield merges in the same way as for pushouts.
def test_coequaliser_counterexample():
    A = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 1, 0],
        ]
    )

    B = np.array(
        [
            [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
        ]
    )

    qmap2 = {6: 0, 10: 1, 14: 2}
    smap2 = {3: 0, 4: 1, 8: 2}

    output_code = coequaliser(A, B, qmap2, smap2, True)
    assert output_code is not None
    assert num_logical_qubits(output_code.Code) == 2  # not 1!
    assert Z_distance(output_code.Code) == 1
    assert len(output_code.NewZLogicals) == 1
    assert len(output_code.NewXLogicals) == 1
    assert len(output_code.OldZLogicals) == 1
    assert len(output_code.OldXLogicals) == 1

    ker_before = np.array(kernel_basis_calc(B)).T
    ker_after = multiply_F2(output_code.MergeMap, ker_before)
    assert compose_to_zero(output_code.Code.PX, ker_after)

    im_before = np.array(image_basis_calc(A)).T
    im_after = multiply_F2(output_code.MergeMap, im_before)
    im_quotient = np.array(
        kernel_basis_calc(np.array(image_basis_calc(output_code.Code.PZ.T)))
    )
    assert compose_to_zero(im_quotient, im_after)
    assert np.array_equal(multiply_F2(output_code.MergeMap, A), output_code.Code.PZ.T)


# [[15, 3, 3]] code
def test_LCS_internal_merge():
    l = 1
    L = 3

    lifted_code = lift_connected_surface_codes(l, L)
    ind1 = [0, 2, 3]
    ind2 = [6, 8, 9]
    merged_code = internal_merge_by_indices(lifted_code, ind1, ind2, "Z", 1, True)
    assert is_valid_CSScode(merged_code.Code)
    assert num_logical_qubits(merged_code.Code) == 2
    assert code_distance(merged_code.Code) == 3

    assert len(merged_code.NewZLogicals) == 0
    assert len(merged_code.NewXLogicals) == 0
    assert len(merged_code.OldZLogicals) == 2
    assert len(merged_code.OldXLogicals) == 2

    ker_before = np.array(kernel_basis_calc(lifted_code.PX)).T
    ker_after = multiply_F2(merged_code.MergeMap, ker_before)
    assert compose_to_zero(merged_code.Code.PX, ker_after)

    im_before = np.array(image_basis_calc(lifted_code.PZ.T)).T
    im_after = multiply_F2(merged_code.MergeMap, im_before)
    im_quotient = np.array(
        kernel_basis_calc(np.array(image_basis_calc(merged_code.Code.PZ.T)))
    )
    assert compose_to_zero(im_quotient, im_after)

    assert num_X_stabs(merged_code.Code) == num_X_stabs(lifted_code)

    assert num_Z_stabs(merged_code.Code) - len(merged_code.NewZStabs) == num_Z_stabs(
        lifted_code
    )

    u = np.array([[1] * (num_data_qubits(lifted_code))]).T
    u_after = multiply_F2(merged_code.MergeMap, u).T[0]

    for i in merged_code.NewQubits:
        assert not u_after[i]

    assert num_data_qubits(merged_code.Code) - len(
        merged_code.NewQubits
    ) == num_data_qubits(lifted_code)


def test_shor_external_merge():
    v = [1, 0, 0, 1, 0, 0, 1, 0, 0]
    indices = vector_to_indices(v)
    indices2 = [i + 9 for i in indices]

    two_Shors = direct_sum_codes(Shor_code, Shor_code)

    merged_code = internal_merge_by_indices(two_Shors, indices, indices2, "Z", 1, True)
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

    assert num_X_stabs(merged_code.Code) == num_X_stabs(two_Shors)

    assert num_Z_stabs(merged_code.Code) - len(merged_code.NewZStabs) == num_Z_stabs(
        two_Shors
    )

    u = np.array([[1] * (num_data_qubits(two_Shors))]).T
    u_after = multiply_F2(merged_code.MergeMap, u).T[0]

    for i in merged_code.NewQubits:
        assert not u_after[i]

    assert num_data_qubits(merged_code.Code) - len(
        merged_code.NewQubits
    ) == num_data_qubits(two_Shors)

    # higher merge depth
    for depth in range(2, 5):
        merged_code = internal_merge_by_indices(
            two_Shors, indices, indices2, "Z", depth, True
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

        assert num_Z_stabs(merged_code.Code) - len(
            merged_code.NewZStabs
        ) == num_Z_stabs(two_Shors)

        assert num_X_stabs(merged_code.Code) - len(
            merged_code.NewXStabs
        ) == num_X_stabs(two_Shors)

        u = np.array([[1] * (num_data_qubits(two_Shors))]).T
        u_after = multiply_F2(merged_code.MergeMap, u).T[0]

        for i in merged_code.NewQubits:
            assert not u_after[i]

        assert num_data_qubits(merged_code.Code) - len(
            merged_code.NewQubits
        ) == num_data_qubits(two_Shors)


def test_shor_surface_external_merge():
    v1 = [1, 0, 0, 1, 0, 0, 1, 0, 0]
    ind1 = vector_to_indices(v1)
    v2 = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ind2 = [i + 9 for i in vector_to_indices(v2)]

    initial_code = direct_sum_codes(Shor_code, Surface3_code)

    merged_code = internal_merge_by_indices(initial_code, ind1, ind2, "Z", 1, True)
    assert merged_code is not None
    assert is_valid_CSScode(merged_code.Code)
    assert num_logical_qubits(merged_code.Code) == 1

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

    # higher merge depth
    for depth in range(2, 5):
        merged_code = internal_merge_by_indices(
            initial_code, ind1, ind2, "Z", depth, True
        )
        assert is_valid_CSScode(merged_code.Code)
        assert num_logical_qubits(merged_code.Code) == 1

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


def test_lifted_external_merge():
    # [[15, 3, 3]] LCS code
    l = 1
    L = 3
    lifted_code = lift_connected_surface_codes(l, L)
    indices1 = [2, 9, 14]
    indices2 = [17, 24, 29]
    combined_code = direct_sum_codes(lifted_code, lifted_code)
    merged_code = internal_merge_by_indices(
        combined_code, indices1, indices2, "Z", 1, True
    )
    assert is_valid_CSScode(merged_code.Code)
    assert num_logical_qubits(merged_code.Code) == 6
    assert len(merged_code.NewZLogicals) == 1
    assert len(merged_code.NewXLogicals) == 1

    merged_code = internal_merge_by_indices(
        combined_code, indices1, indices2, "Z", 1, False
    )
    assert is_valid_CSScode(merged_code)
    assert num_logical_qubits(merged_code) == 6

    flipped_code = flip_code(combined_code)
    merged_code = internal_merge_by_indices(
        flipped_code, indices1, indices2, "X", 1, False
    )
    assert is_valid_CSScode(merged_code)
    assert num_logical_qubits(merged_code) == 6

    merged_code = internal_merge_by_indices(
        flipped_code, indices1, indices2, "X", 1, True
    )
    assert is_valid_CSScode(merged_code.Code)
    assert num_logical_qubits(merged_code.Code) == 6
    assert len(merged_code.NewZLogicals) == 1
    assert len(merged_code.NewXLogicals) == 1


def test_logical_splitting():
    # [[15, 3, 3]] LCS code
    l = 1
    L = 3
    lifted_code = lift_connected_surface_codes(l, L)
    indices1 = [2, 9, 14]
    indices2 = [17, 24, 29]
    combined_code = direct_sum_codes(lifted_code, lifted_code)
    merged_code = internal_merge_by_indices(
        combined_code, indices1, indices2, "Z", 1, True
    )

    num_before_logicals = 6
    num_after_logicals = 6

    with pytest.raises(ValueError, match="Must enter Z or X basis."):
        find_logical_splitting(
            combined_code,
            merged_code.Code,
            merged_code.MergeMap,
            num_before_logicals,
            num_after_logicals,
            num_merges=1,
            basis="Y",
        )

    flipped_code = flip_code(combined_code)
    merged_code = internal_merge_by_indices(
        flipped_code, indices1, indices2, "X", 1, True
    )
    num_before_logicals = 7
    with pytest.raises(
        RuntimeError, match="Dimensions do not match up when returning merge data."
    ):
        find_logical_splitting(
            flipped_code,
            merged_code.Code,
            merged_code.MergeMap,
            num_before_logicals,
            num_after_logicals,
            num_merges=1,
            basis="X",
        )

    num_before_logicals = 6
    (new_Z_logicals, new_X_logicals, old_Z_logicals, old_X_logicals) = (
        find_logical_splitting(
            flipped_code,
            merged_code.Code,
            merged_code.MergeMap,
            num_before_logicals,
            num_after_logicals,
            num_merges=1,
            basis="X",
        )
    )
    assert len(new_Z_logicals) == 1
    assert len(new_X_logicals) == 1
    assert len(old_Z_logicals) == 5
    assert len(old_X_logicals) == 5
