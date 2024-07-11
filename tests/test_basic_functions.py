from itertools import combinations

import numpy as np
import pytest

from ssip.basic_functions import (
    CSScode,
    compose_to_zero,
    dot_product,
    find_homology_basis,
    find_paired_basis,
    find_Z_basis,
    image_all_vectors,
    image_basis_calc,
    indices_to_vector,
    is_valid_CSScode,
    kernel_all_vectors,
    kernel_basis_calc,
    max_measurement_support,
    max_measurement_weight,
    num_logical_qubits,
    vec_addition,
)
from ssip.code_examples import (
    LIST_ALL_CODES,
    QRM_A,
    QRM_B,
    GTri_A,
    GTri_B,
    GTri_code,
    GTri_X_basis,
    MergedShor_A,
    MergedShor_B,
    MergedShor_code,
    QRM_code,
    Shor_A,
    Shor_B,
    Shor_code,
    Steane_A,
    Steane_B,
    Steane_code,
    Surface2_A,
    Surface2_B,
    Surface2_code,
    Surface3_A,
    Surface3_B,
    Surface3_code,
    Toric_A,
    Toric_B,
    Toric_code,
    Twist_A,
    Twist_B,
    Twist_code,
)


# helper function, raises an error in the *correct* case,
# where none of the homology basis vectors sum to give
# a vector in the [0] equivalence class, i.e. they are
# all linearly independent.
def check_hom_basis(hom_basis, A):
    ims = image_all_vectors(A)
    all_combs = []
    for i in range(1, len(hom_basis) + 1):
        all_combs += list(combinations(hom_basis, i))
    for i in range(len(all_combs)):
        all_combs[i] = vec_addition(all_combs[i])
    for i in ims:
        all_combs.index(i)


##### Tests #####


def test_vec_addition():
    testvec1 = [0, 0, 1, 1]
    testvec2 = [0, 1, 0, 1]
    vecs = [testvec1, testvec2]
    assert vec_addition(vecs) == [0, 1, 1, 0]


def test_indices_to_vector():
    v = indices_to_vector([0], 1)
    assert v == [1]

    v = indices_to_vector([0, 2, 3], 5)
    assert v == [1, 0, 1, 1, 0]


def test_dot_product():
    testvec1 = [0, 0, 1, 1]
    testvec2 = [0, 1, 0, 1]
    assert dot_product(testvec1, testvec2) == 1
    with pytest.raises(ValueError):
        dot_product([0, 1], [0, 0, 0])


def test_composition():
    assert compose_to_zero(Shor_B, Shor_A)
    assert compose_to_zero(MergedShor_B, MergedShor_A)
    assert compose_to_zero(GTri_B, GTri_A)
    assert compose_to_zero(Toric_B, Toric_A)
    assert compose_to_zero(QRM_B, QRM_A)
    assert compose_to_zero(Steane_B, Steane_A)
    assert compose_to_zero(Surface3_B, Surface3_A)
    assert compose_to_zero(Surface2_B, Surface2_A)

    wrong_A = np.array([[1]])
    wrong_B = np.array([[1]])
    assert not compose_to_zero(wrong_B, wrong_A)
    wrong_code = CSScode(wrong_A, wrong_B)
    assert not is_valid_CSScode(wrong_code)

    for code in LIST_ALL_CODES:
        assert is_valid_CSScode(code)


def test_weight():
    assert max_measurement_support(Shor_code) == 2
    assert max_measurement_weight(Shor_code) == 6
    assert max_measurement_support(Surface3_code) == 2
    assert max_measurement_weight(Surface3_code) == 4


def test_kernel():
    kers = kernel_basis_calc(Shor_B)
    assert len(kers) == 7
    for k in kers:
        arr = np.array([k]).T
        assert compose_to_zero(Shor_B, arr)

    all_kers = kernel_all_vectors(Shor_B)
    assert len(all_kers) == 2 ** len(kers) - 1

    kers = kernel_basis_calc(Shor_A)
    assert len(kers) == 0
    for k in kers:
        arr = np.array([k]).T
        assert compose_to_zero(Shor_A, arr)

    all_kers = kernel_all_vectors(Shor_A)
    assert len(all_kers) == 0

    kers = kernel_basis_calc(MergedShor_B)
    for k in kers:
        arr = np.array([k]).T
        assert compose_to_zero(MergedShor_B, arr)

    all_kers = kernel_all_vectors(MergedShor_B)
    assert len(all_kers) == 2 ** len(kers) - 1

    kers = kernel_basis_calc(MergedShor_A)
    for k in kers:
        arr = np.array([k]).T
        assert compose_to_zero(MergedShor_A, arr)

    all_kers = kernel_all_vectors(MergedShor_A)
    assert len(all_kers) == 2 ** len(kers) - 1

    kers = kernel_basis_calc(GTri_B)
    for k in kers:
        arr = np.array([k]).T
        assert compose_to_zero(GTri_B, arr)

    all_kers = kernel_all_vectors(GTri_B)
    assert len(all_kers) == 2 ** len(kers) - 1

    kers = kernel_basis_calc(GTri_A)
    for k in kers:
        arr = np.array([k]).T
        assert compose_to_zero(GTri_A, arr)

    all_kers = kernel_all_vectors(GTri_A)
    assert len(all_kers) == 2 ** len(kers) - 1

    kers = kernel_basis_calc(Toric_B)
    for k in kers:
        arr = np.array([k]).T
        assert compose_to_zero(Toric_B, arr)

    all_kers = kernel_all_vectors(Toric_B)
    assert len(all_kers) == 2 ** len(kers) - 1

    kers = kernel_basis_calc(Toric_A)
    for k in kers:
        arr = np.array([k]).T
        assert compose_to_zero(Toric_A, arr)

    all_kers = kernel_all_vectors(Toric_A)
    assert len(all_kers) == 2 ** len(kers) - 1

    kers = kernel_basis_calc(QRM_B)
    for k in kers:
        arr = np.array([k]).T
        assert compose_to_zero(QRM_B, arr)

    all_kers = kernel_all_vectors(QRM_B)
    assert len(all_kers) == 2 ** len(kers) - 1


def test_image():
    ims = image_basis_calc(Shor_A)
    assert len(ims) == 6
    for i in ims:
        arr = np.array([i]).T
        assert compose_to_zero(Shor_B, arr)

    all_ims = image_all_vectors(Shor_A)
    assert len(all_ims) == 2 ** len(ims) - 1

    ims = image_basis_calc(Shor_B)
    assert len(ims) == 2

    all_ims = image_all_vectors(Shor_B)
    assert len(all_ims) == 2 ** len(ims) - 1

    ims = image_basis_calc(Toric_A)
    assert len(ims) == 8

    all_ims = image_all_vectors(Toric_A)
    assert len(all_ims) == 2 ** len(ims) - 1


def test_homology():
    hom_basis = find_homology_basis(Shor_A, Shor_B)
    assert num_logical_qubits(Shor_code) == 1
    assert len(hom_basis) == 1
    assert sum(hom_basis[0]) >= 3
    with pytest.raises(ValueError):
        check_hom_basis(hom_basis, Shor_A)

    new_mat = np.array([[1, 0], [0, 1]])
    new_mat2 = np.array([[1, 1], [1, 1]])
    with pytest.raises(ValueError):
        find_homology_basis(new_mat, new_mat2)

    Z_basis = find_Z_basis(Shor_code)
    assert len(Z_basis) == 1
    assert sum(Z_basis[0]) >= 3
    with pytest.raises(ValueError):
        check_hom_basis(Z_basis, Shor_A)

    hom_basis = find_homology_basis(MergedShor_A, MergedShor_B)
    assert num_logical_qubits(MergedShor_code) == 1
    assert len(hom_basis) == 1
    assert sum(hom_basis[0]) >= 3
    with pytest.raises(ValueError):
        check_hom_basis(hom_basis, MergedShor_A)

    Z_basis = find_Z_basis(MergedShor_code)
    assert len(Z_basis) == 1
    assert sum(Z_basis[0]) >= 3
    with pytest.raises(ValueError):
        check_hom_basis(Z_basis, Shor_A)

    hom_basis = find_homology_basis(GTri_A, GTri_B)
    assert num_logical_qubits(GTri_code) == 3
    assert len(hom_basis) == 3
    for b in hom_basis:
        assert sum(b) >= 2
    with pytest.raises(ValueError):
        check_hom_basis(hom_basis, GTri_A)

    Z_basis = find_Z_basis(GTri_code)
    assert len(Z_basis) == 3
    assert sum(Z_basis[0]) >= 2
    with pytest.raises(ValueError):
        check_hom_basis(Z_basis, GTri_A)

    hom_basis = find_homology_basis(Toric_A, Toric_B)
    assert num_logical_qubits(Toric_code) == 2
    assert len(hom_basis) == 2
    for b in hom_basis:
        assert sum(b) >= 3
    with pytest.raises(ValueError):
        check_hom_basis(hom_basis, Toric_A)

    hom_basis = find_homology_basis(QRM_A, QRM_B)
    assert num_logical_qubits(QRM_code) == 1
    assert len(hom_basis) == 1
    assert sum(hom_basis[0]) >= 3

    hom_basis = find_homology_basis(Twist_A, Twist_B)
    assert num_logical_qubits(Twist_code) == 2
    assert len(hom_basis) == 2
    for b in hom_basis:
        assert sum(b) >= 3
    with pytest.raises(ValueError):
        check_hom_basis(hom_basis, Twist_A)

    hom_basis = find_homology_basis(Steane_A, Steane_B)
    assert num_logical_qubits(Steane_code) == 1
    assert len(hom_basis) == 1
    assert sum(hom_basis[0]) >= 3

    hom_basis = find_homology_basis(Surface3_A, Surface3_B)
    assert num_logical_qubits(Surface3_code) == 1
    assert len(hom_basis) == 1
    assert sum(hom_basis[0]) >= 3

    hom_basis = find_homology_basis(Surface2_A, Surface2_B)
    assert num_logical_qubits(Surface2_code) == 1
    assert len(hom_basis) == 1
    assert sum(hom_basis[0]) >= 2


def test_paired_basis():
    bA = [0, 1]
    bB = [0]
    with pytest.raises(ValueError):
        find_paired_basis(bA, bB)

    hom_dim = num_logical_qubits(GTri_code)
    hom_basis = find_homology_basis(GTri_A, GTri_B)
    hom_basis2 = find_homology_basis(GTri_B.T, GTri_A.T)
    replacement_basis2 = find_paired_basis(hom_basis, hom_basis2)
    for b in hom_basis:
        overlap_one = 0
        overlap_zero = 0
        for b2 in replacement_basis2:
            if dot_product(b, b2):
                overlap_one += 1
            else:
                overlap_zero += 1
        assert overlap_one == 1
        assert overlap_zero == hom_dim - 1

    hom_basis3 = list(
        GTri_X_basis
    )  # Try with one pre-determined valid choice of X basis
    replacement_basis2 = find_paired_basis(hom_basis3, hom_basis)
    for b in hom_basis3:
        overlap_one = 0
        overlap_zero = 0
        for b2 in replacement_basis2:
            if dot_product(b, b2):
                overlap_one += 1
            else:
                overlap_zero += 1
        assert overlap_one == 1
        assert overlap_zero == hom_dim - 1

    # other way round
    replacement_basis2 = find_paired_basis(hom_basis2, hom_basis)
    for b in hom_basis2:
        overlap_one = 0
        overlap_zero = 0
        for b2 in replacement_basis2:
            if dot_product(b, b2):
                overlap_one += 1
            else:
                overlap_zero += 1
        assert overlap_one == 1
        assert overlap_zero == hom_dim - 1

    hom_dim = num_logical_qubits(Toric_code)
    hom_basis = find_homology_basis(Toric_A, Toric_B)
    hom_basis2 = find_homology_basis(Toric_B.T, Toric_A.T)
    replacement_basis2 = find_paired_basis(hom_basis, hom_basis2)
    for b in hom_basis:
        overlap_one = 0
        overlap_zero = 0
        for b2 in replacement_basis2:
            if dot_product(b, b2):
                overlap_one += 1
            else:
                overlap_zero += 1
        assert overlap_one == 1
        assert overlap_zero == hom_dim - 1

    # other way round
    replacement_basis2 = find_paired_basis(hom_basis2, hom_basis)
    for b in hom_basis2:
        overlap_one = 0
        overlap_zero = 0
        for b2 in replacement_basis2:
            if dot_product(b, b2):
                overlap_one += 1
            else:
                overlap_zero += 1
        assert overlap_one == 1
        assert overlap_zero == hom_dim - 1

    hom_dim = num_logical_qubits(Twist_code)
    hom_basis = find_homology_basis(Twist_A, Twist_B)
    hom_basis2 = find_homology_basis(Twist_B.T, Twist_A.T)
    replacement_basis2 = find_paired_basis(hom_basis, hom_basis2)
    for b in hom_basis:
        overlap_one = 0
        overlap_zero = 0
        for b2 in replacement_basis2:
            if dot_product(b, b2):
                overlap_one += 1
            else:
                overlap_zero += 1
        assert overlap_one == 1
        assert overlap_zero == hom_dim - 1

    # other way round
    replacement_basis2 = find_paired_basis(hom_basis2, hom_basis)
    for b in hom_basis2:
        overlap_one = 0
        overlap_zero = 0
        for b2 in replacement_basis2:
            if dot_product(b, b2):
                overlap_one += 1
            else:
                overlap_zero += 1
        assert overlap_one == 1
        assert overlap_zero == hom_dim - 1
