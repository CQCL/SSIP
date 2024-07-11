from pathlib import Path

from ssip.auto_surgeries import measure_single_logical_qubit
from ssip.basic_functions import (
    dot_product,
    indices_to_vector,
    is_valid_CSScode,
    max_measurement_support,
    max_measurement_weight,
    num_data_qubits,
    num_logical_qubits,
    vector_to_indices,
)
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
)

l = 12
m = 6
powers_A = ([3], [1, 2])
powers_B = ([1, 2], [3])

bicycle_code = bivariate_bicycle_code(l, m, powers_A, powers_B)
n = num_data_qubits(bicycle_code)
k = num_logical_qubits(bicycle_code)
initial_d = 12


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

num_before_stabilisers = bicycle_code.PX.shape[0] + bicycle_code.PZ.shape[0]

print("----Xs----")
with Path.open("results/gross_X_singleqs.txt", "w") as f:
    f.write("X measurements:\n")
    for i in range(k):
        print("logical: " + str(i))
        ind = vector_to_indices(X_logicals[i])
        r = 0
        while True:
            r += 1
            print("r = ", r)
            pushout_data = measure_single_logical_qubit(bicycle_code, ind, "X", r, True)

            assert is_valid_CSScode(pushout_data.Code)
            sd = subsystem_distance_GAP(
                pushout_data.Code,
                pushout_data.NewZLogicals,
                pushout_data.NewXLogicals,
                None,
                10000,
            )
            if sd >= initial_d:
                print("sd >= initial_d")
                print("r = ", r)

                n_new = num_data_qubits(pushout_data.Code) - num_data_qubits(
                    bicycle_code
                )
                frac = n_new / (num_data_qubits(bicycle_code))
                print(frac)
                print("num qubits = ", num_data_qubits(pushout_data.Code))
                omega = max(
                    max_measurement_support(pushout_data.Code),
                    max_measurement_weight(pushout_data.Code),
                )
                print("omega = ", omega)
                num_after_stabilisers = (
                    pushout_data.Code.PX.shape[0] + pushout_data.Code.PZ.shape[0]
                )
                print(num_after_stabilisers - num_before_stabilisers)
                f.write(
                    str(i)
                    + ","
                    + str(r)
                    + ","
                    + str(num_data_qubits(pushout_data.Code))
                    + ","
                    + str(frac)
                    + ","
                    + str(omega)
                    + "\n"
                )
                break

print("----Zs----")
with Path.open("results/gross_Z_singleqs.txt", "w") as f:
    f.write("Z measurements:\n")
    for i in range(k):
        print("logical: " + str(i))
        ind = vector_to_indices(Z_logicals[i])
        r = 0
        while True:
            r += 1
            print("r = ", r)
            pushout_data = measure_single_logical_qubit(bicycle_code, ind, "Z", r, True)

            assert is_valid_CSScode(pushout_data.Code)
            sd = subsystem_distance_GAP(
                pushout_data.Code,
                pushout_data.NewZLogicals,
                pushout_data.NewXLogicals,
                None,
                10000,
            )
            if sd >= initial_d:
                print("sd >= initial_d")
                print("r = ", r)

                n_new = num_data_qubits(pushout_data.Code) - num_data_qubits(
                    bicycle_code
                )
                frac = n_new / (num_data_qubits(bicycle_code))
                print(frac)
                print("num qubits = ", num_data_qubits(pushout_data.Code))
                omega = max(
                    max_measurement_support(pushout_data.Code),
                    max_measurement_weight(pushout_data.Code),
                )
                print("omega = ", omega)
                num_after_stabilisers = (
                    pushout_data.Code.PX.shape[0] + pushout_data.Code.PZ.shape[0]
                )
                print(num_after_stabilisers - num_before_stabilisers)
                f.write(
                    str(i)
                    + ","
                    + str(r)
                    + ","
                    + str(num_data_qubits(pushout_data.Code))
                    + ","
                    + str(frac)
                    + ","
                    + str(omega)
                    + "\n"
                )
                break
