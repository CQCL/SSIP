from pathlib import Path

import numpy as np

from ssip.basic_functions import (
    compose_to_zero,
    dot_product,
    image_basis_calc,
    indices_to_vector,
    kernel_basis_calc,
    max_measurement_support,
    max_measurement_weight,
    multiply_F2,
    num_data_qubits,
    num_logical_qubits,
    vector_to_indices,
)
from ssip.coequalisers import internal_merge_by_indices
from ssip.distance import subsystem_distance_GAP
from ssip.lifted_product import (
    bivariate_bicycle_code,
    primed_X_logical,
    primed_Z_logical,
    unprimed_X_logical,
    unprimed_Z_logical,
)
from ssip.monic_span_checker import (
    is_irreducible_logical,
    monic_span,
    restricted_matrix,
)

l = 12
m = 6
powers_A = ([3], [1, 2])
powers_B = ([1, 2], [3])

bicycle_code = bivariate_bicycle_code(l, m, powers_A, powers_B)
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
    X_logicals.append(indices_to_vector(X_logical, n))

for alpha in alphas_n:
    X_logical = primed_X_logical(alpha, g, h, l, m)
    assert is_irreducible_logical(X_logical, bicycle_code.PZ)
    X_logicals.append(indices_to_vector(X_logical, n))

for alpha in alphas_m:
    Z_logical = unprimed_Z_logical(alpha, h, g, l, m)
    assert is_irreducible_logical(Z_logical, bicycle_code.PX)
    Z_logicals.append(indices_to_vector(Z_logical, n))

for alpha in alphas_m:
    Z_logical = primed_Z_logical(alpha, f, l, m)
    assert is_irreducible_logical(Z_logical, bicycle_code.PX)
    Z_logicals.append(indices_to_vector(Z_logical, n))

for i in range(k):
    for j in range(k):
        if i == j:
            assert dot_product(X_logicals[i], Z_logicals[j]) == 1
        else:
            assert dot_product(X_logicals[i], Z_logicals[j]) == 0

with Path.open("results/gross_int_X_merges.txt", "w") as f:
    for i in range(k):
        for j in range(i):
            indices1 = vector_to_indices(X_logicals[i])
            indices2 = vector_to_indices(X_logicals[j])
            restr1 = restricted_matrix(indices1, bicycle_code.PZ)
            restr2 = restricted_matrix(indices2, bicycle_code.PZ)

            span = monic_span(restr1[0], restr2[0])
            if (i < 6 and j < 6) or (i > 5 and j > 5):
                print(str(i) + "," + str(j))
                assert span is not None
                duplicate = False
                for ind in indices1:
                    if ind in indices2:
                        duplicate = True
                        break
                print("duplicate qubits = " + str(duplicate))
                if not duplicate:
                    syndrome_map = {}
                    for syndrome1, syndrome2 in span[1].items():
                        syndrome_map[restr1[1][syndrome1]] = restr2[1][syndrome2]
                    for syndrome2 in syndrome_map.values():
                        if syndrome2 in syndrome_map:
                            duplicate = True
                            print("duplicate Z-checks")
                            break
                    if not duplicate:
                        depth = 1
                        while depth:
                            print("depth = ", depth)
                            merged_code = internal_merge_by_indices(
                                bicycle_code, indices1, indices2, "X", depth, True
                            )
                            ker_before = np.array(kernel_basis_calc(bicycle_code.PZ)).T
                            ker_after = multiply_F2(merged_code.MergeMap, ker_before)
                            assert compose_to_zero(merged_code.Code.PZ, ker_after)

                            im_before = np.array(image_basis_calc(bicycle_code.PX.T)).T
                            im_after = multiply_F2(merged_code.MergeMap, im_before)
                            im_quotient = np.array(
                                kernel_basis_calc(
                                    np.array(image_basis_calc(merged_code.Code.PX.T))
                                )
                            )
                            assert compose_to_zero(im_quotient, im_after)
                            sd = subsystem_distance_GAP(
                                merged_code.Code,
                                merged_code.NewZLogicals,
                                merged_code.NewXLogicals,
                                None,
                            )
                            print(sd)
                            if sd >= 12:
                                n_new = len(merged_code.NewQubits)
                                print(n_new)
                                frac = n_new / 144
                                omega = max(
                                    max_measurement_support(merged_code.Code),
                                    max_measurement_weight(merged_code.Code),
                                )
                                f.write(
                                    str(i)
                                    + ","
                                    + str(j)
                                    + ","
                                    + str(depth)
                                    + ","
                                    + str(frac)
                                    + ","
                                    + str(omega)
                                    + "\n"
                                )
                                break
                            else:
                                depth += 1

            else:
                assert span is None


with Path.open("results/gross_int_Z_merges.txt", "w") as f:
    for i in range(k):
        for j in range(i):
            indices1 = vector_to_indices(Z_logicals[i])
            indices2 = vector_to_indices(Z_logicals[j])
            restr1 = restricted_matrix(indices1, bicycle_code.PX)
            restr2 = restricted_matrix(indices2, bicycle_code.PX)

            span = monic_span(restr1[0], restr2[0])
            if (i < 6 and j < 6) or (i > 5 and j > 5):
                print(str(i) + "," + str(j))
                assert span is not None
                duplicate = False
                for ind in indices1:
                    if ind in indices2:
                        duplicate = True
                        break
                print("duplicate qubits = " + str(duplicate))
                if not duplicate:
                    syndrome_map = {}
                    for syndrome1, syndrome2 in span[1].items():
                        syndrome_map[restr1[1][syndrome1]] = restr2[1][syndrome2]
                    for syndrome2 in syndrome_map.values():
                        if syndrome2 in syndrome_map:
                            duplicate = True
                            print("duplicate X-checks")
                            break
                    if not duplicate:
                        depth = 1
                        while depth:
                            print("depth = ", depth)
                            merged_code = internal_merge_by_indices(
                                bicycle_code, indices1, indices2, "Z", depth, True
                            )
                            ker_before = np.array(kernel_basis_calc(bicycle_code.PX)).T
                            ker_after = multiply_F2(merged_code.MergeMap, ker_before)
                            assert compose_to_zero(merged_code.Code.PX, ker_after)

                            im_before = np.array(image_basis_calc(bicycle_code.PZ.T)).T
                            im_after = multiply_F2(merged_code.MergeMap, im_before)
                            im_quotient = np.array(
                                kernel_basis_calc(
                                    np.array(image_basis_calc(merged_code.Code.PZ.T))
                                )
                            )
                            assert compose_to_zero(im_quotient, im_after)
                            num = num_data_qubits(merged_code.Code)
                            sd = subsystem_distance_GAP(
                                merged_code.Code,
                                merged_code.NewZLogicals,
                                merged_code.NewXLogicals,
                                None,
                            )
                            print(sd)
                            if sd >= 12:
                                n_new = len(merged_code.NewQubits)
                                print(n_new)
                                frac = n_new / 144
                                omega = max(
                                    max_measurement_support(merged_code.Code),
                                    max_measurement_weight(merged_code.Code),
                                )
                                f.write(
                                    str(i)
                                    + ","
                                    + str(j)
                                    + ","
                                    + str(depth)
                                    + ","
                                    + str(frac)
                                    + ","
                                    + str(omega)
                                    + "\n"
                                )
                                break
                            else:
                                depth += 1

            else:
                assert span is None
