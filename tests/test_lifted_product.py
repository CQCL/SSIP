import numpy as np

from ssip.basic_functions import (
    compose_to_zero,
    dot_product,
    indices_to_vector,
    is_valid_CSScode,
    num_data_qubits,
    num_logical_qubits,
    vector_to_indices,
)
from ssip.distance import code_distance, distance_GAP
from ssip.lifted_product import (
    bivariate_bicycle_code,
    generalised_bicycle_code,
    lift_connected_surface_codes,
    primed_X_logical,
    primed_Z_logical,
    tensor_product,
    unprimed_X_logical,
    unprimed_Z_logical,
)
from ssip.monic_span_checker import (
    is_irreducible_logical,
    monic_span,
    restricted_matrix,
)


def test_tensor_product():
    A = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    toric = tensor_product(A, A)
    assert is_valid_CSScode(toric)
    assert num_data_qubits(toric) == 18
    assert num_logical_qubits(toric) == 2
    assert code_distance(toric) == 3


def test_lift_connected_surface_codes():
    # [[10, 2, 2]] LCS
    l = 1
    L = 2

    lifted_code = lift_connected_surface_codes(l, L)
    assert is_valid_CSScode(lifted_code)
    assert num_data_qubits(lifted_code) == 10
    assert num_logical_qubits(lifted_code) == 2
    assert code_distance(lifted_code) == 2

    # [[15, 3, 3]] LCS
    l = 1
    L = 3

    lifted_code = lift_connected_surface_codes(l, L)
    assert is_valid_CSScode(lifted_code)
    assert num_data_qubits(lifted_code) == 15
    assert num_logical_qubits(lifted_code) == 3
    assert code_distance(lifted_code) == 3

    # [[20, 4, 3]] LCS
    l = 1
    L = 4
    lifted_code = lift_connected_surface_codes(l, L)
    assert is_valid_CSScode(lifted_code)
    assert num_data_qubits(lifted_code) == 20
    assert num_logical_qubits(lifted_code) == 4
    assert code_distance(lifted_code) == 3

    # [[26, 2, 2]] LCS
    l = 2
    L = 2
    lifted_code = lift_connected_surface_codes(l, L)
    assert is_valid_CSScode(lifted_code)
    assert num_data_qubits(lifted_code) == 26
    assert num_logical_qubits(lifted_code) == 2

    # [[175, 7, 7]] LCS
    l = 3
    L = 7
    lifted_code = lift_connected_surface_codes(l, L)
    assert is_valid_CSScode(lifted_code)
    assert num_data_qubits(lifted_code) == 175
    assert num_logical_qubits(lifted_code) == 7


def test_bivariant_bicycle_codes():
    # [[72, 12, 6]] quasi-abelian code
    l = 6
    m = 6
    powers_A = ([3], [1, 2])
    powers_B = ([1, 2], [3])

    bicycle_code = bivariate_bicycle_code(l, m, powers_A, powers_B)
    assert is_valid_CSScode(bicycle_code)
    assert num_data_qubits(bicycle_code) == 72
    assert num_logical_qubits(bicycle_code) == 12

    # [[90, 8, 10]] quasi-abelian code
    l = 15
    m = 3
    powers_A = ([9], [1, 2])
    powers_B = ([0, 2, 7], [])

    bicycle_code = bivariate_bicycle_code(l, m, powers_A, powers_B)
    assert is_valid_CSScode(bicycle_code)
    assert num_data_qubits(bicycle_code) == 90
    assert num_logical_qubits(bicycle_code) == 8

    # [[144, 12, 12]] quasi-abelian code
    l = 12
    m = 6
    powers_A = ([3], [1, 2])
    powers_B = ([1, 2], [3])

    bicycle_code = bivariate_bicycle_code(l, m, powers_A, powers_B)
    assert is_valid_CSScode(bicycle_code)
    assert num_data_qubits(bicycle_code) == 144
    assert num_logical_qubits(bicycle_code) == 12


def test_generalised_bicycle_codes():
    # [[48, 6, 8]]
    l = 24
    powers_A = [0, 2, 8, 15]
    powers_B = [0, 2, 12, 17]

    GB_code = generalised_bicycle_code(l, powers_A, powers_B)
    assert is_valid_CSScode(GB_code)
    assert num_data_qubits(GB_code) == 48
    assert num_logical_qubits(GB_code) == 6

    # [[126, 28, 8]]
    l = 63
    powers_A = [0, 1, 14, 16, 22]
    powers_B = [0, 3, 13, 20, 42]

    GB_code = generalised_bicycle_code(l, powers_A, powers_B)
    assert is_valid_CSScode(GB_code)
    assert num_data_qubits(GB_code) == 126
    assert num_logical_qubits(GB_code) == 28

    # [[46, 2, 9]]
    l = 23
    powers_A = [0, 5, 8, 12]
    powers_B = [0, 1, 5, 7]

    GB_code = generalised_bicycle_code(l, powers_A, powers_B)
    assert is_valid_CSScode(GB_code)
    assert num_data_qubits(GB_code) == 46
    assert num_logical_qubits(GB_code) == 2

    # [[180, 10, d]]
    l = 90
    powers_A = [0, 28, 80, 89]
    powers_B = [0, 2, 21, 25]

    GB_code = generalised_bicycle_code(l, powers_A, powers_B)
    assert is_valid_CSScode(GB_code)
    assert num_data_qubits(GB_code) == 180
    assert num_logical_qubits(GB_code) == 10

    # [[254, 28, d]]
    l = 127
    powers_A = [0, 15, 20, 28, 66]
    powers_B = [0, 58, 59, 100, 121]

    GB_code = generalised_bicycle_code(l, powers_A, powers_B)
    assert is_valid_CSScode(GB_code)
    assert num_data_qubits(GB_code) == 254
    assert num_logical_qubits(GB_code) == 28


# See https://arxiv.org/abs/2308.07915 Sec 9.1
def test_logical_pauli_polynomials():
    l = 12
    m = 6
    powers_A = ([3], [1, 2])
    powers_B = ([1, 2], [3])

    bicycle_code = bivariate_bicycle_code(l, m, powers_A, powers_B)
    assert distance_GAP(bicycle_code) == 12
    n = num_data_qubits(bicycle_code)
    k = num_logical_qubits(bicycle_code)

    f = [
        (0, 0),
        (1, 0),
        (1, 3),
        (2, 0),
        (3, 0),
        (5, 3),
        (6, 0),
        (7, 0),
        (7, 3),
        (8, 0),
        (9, 0),
        (11, 3),
    ]
    g = [(0, 2), (0, 4), (1, 0), (1, 2), (2, 1), (2, 3)]
    h = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 3)]

    alphas_n = [(0, 0), (0, 1), (2, 1), (2, 5), (3, 2), (4, 0)]
    alphas_m = [(0, 1), (0, 5), (1, 1), (0, 0), (4, 0), (5, 2)]

    X_logicals = []
    Z_logicals = []

    for alpha in alphas_n:
        X_logical = unprimed_X_logical(alpha, f, l, m)
        assert is_irreducible_logical(X_logical, bicycle_code.PZ)
        X_logical = indices_to_vector(X_logical, n)
        assert compose_to_zero(bicycle_code.PZ, np.array([X_logical]).T)
        X_logicals.append(X_logical)

    for alpha in alphas_n:
        X_logical = primed_X_logical(alpha, g, h, l, m)
        assert is_irreducible_logical(X_logical, bicycle_code.PZ)
        X_logical = indices_to_vector(X_logical, n)
        assert compose_to_zero(bicycle_code.PZ, np.array([X_logical]).T)
        X_logicals.append(X_logical)

    for alpha in alphas_m:
        Z_logical = unprimed_Z_logical(alpha, h, g, l, m)
        assert is_irreducible_logical(Z_logical, bicycle_code.PX)
        Z_logical = indices_to_vector(Z_logical, n)
        assert compose_to_zero(bicycle_code.PX, np.array([Z_logical]).T)
        Z_logicals.append(Z_logical)

    for alpha in alphas_m:
        Z_logical = primed_Z_logical(alpha, f, l, m)
        assert is_irreducible_logical(Z_logical, bicycle_code.PX)
        Z_logical = indices_to_vector(Z_logical, n)
        assert compose_to_zero(bicycle_code.PX, np.array([Z_logical]).T)
        Z_logicals.append(Z_logical)

    for i in range(k):
        for j in range(k):
            if i == j:
                assert dot_product(X_logicals[i], Z_logicals[j]) == 1
            else:
                assert dot_product(X_logicals[i], Z_logicals[j]) == 0

    # check that logicals in the unprimed block match up, and the same for primed.
    for i in range(k):
        for j in range(i):
            indices1 = vector_to_indices(X_logicals[i])
            indices2 = vector_to_indices(X_logicals[j])
            restr1 = restricted_matrix(indices1, bicycle_code.PZ)
            restr2 = restricted_matrix(indices2, bicycle_code.PZ)

            span = monic_span(restr1[0], restr2[0])
            if (i < 6 and j < 6) or (i > 5 and j > 5):
                assert span is not None
            else:
                assert span is None

    for i in range(k):
        for j in range(i):
            indices1 = vector_to_indices(Z_logicals[i])
            indices2 = vector_to_indices(Z_logicals[j])
            restr1 = restricted_matrix(indices1, bicycle_code.PX)
            restr2 = restricted_matrix(indices2, bicycle_code.PX)

            span = monic_span(restr1[0], restr2[0])
            if (i < 6 and j < 6) or (i > 5 and j > 5):
                assert span is not None
            else:
                assert span is None
