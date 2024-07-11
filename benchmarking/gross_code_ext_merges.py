from pathlib import Path

from ssip.basic_functions import (
    indices_to_vector,
    max_measurement_support,
    max_measurement_weight,
    num_data_qubits,
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
from ssip.pushouts import external_merge_by_indices

l = 12
m = 6
powers_A = ([3], [1, 2])
powers_B = ([1, 2], [3])

bicycle_code = bivariate_bicycle_code(l, m, powers_A, powers_B)
n = 144
k = 12
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

print("----Xs----")
for alpha in alphas_n:
    X_logical = unprimed_X_logical(alpha, f, l, m)
    X_logicals.append(indices_to_vector(X_logical, n))

for alpha in alphas_n:
    X_logical = primed_X_logical(alpha, g, h, l, m)
    X_logicals.append(indices_to_vector(X_logical, n))

for alpha in alphas_m:
    Z_logical = unprimed_Z_logical(alpha, h, g, l, m)
    Z_logicals.append(indices_to_vector(Z_logical, n))

for alpha in alphas_m:
    Z_logical = primed_Z_logical(alpha, f, l, m)
    Z_logicals.append(indices_to_vector(Z_logical, n))

with Path.open("results/gross_ext_X_merges.txt", "w") as f:
    f.write("X merges:\n")
    for i in range(k):
        for j in range(i + 1):
            indices1 = vector_to_indices(X_logicals[i])
            indices2 = vector_to_indices(X_logicals[j])

            if (i < 6 and j < 6) or (i > 5 and j > 5):
                print(str(i) + "," + str(j))
                r = 0
                while True:
                    r += 1
                    print("r = ", r)
                    merge_data = external_merge_by_indices(
                        bicycle_code, bicycle_code, indices1, indices2, "X", r, True
                    )
                    sd = subsystem_distance_GAP(
                        merge_data.Code,
                        merge_data.NewZLogicals,
                        merge_data.NewXLogicals,
                        None,
                    )
                    if sd >= initial_d:
                        print("sd >= initial_d")
                        n_new = len(merge_data.NewQubits)
                        frac = n_new / (2 * n)
                        print(frac)
                        print("num qubits = ", num_data_qubits(merge_data.Code))
                        omega = max(
                            max_measurement_support(merge_data.Code),
                            max_measurement_weight(merge_data.Code),
                        )
                        print("omega = ", omega)
                        f.write(
                            str(i)
                            + ","
                            + str(j)
                            + ","
                            + str(r)
                            + ","
                            + str(frac)
                            + ","
                            + str(omega)
                            + ","
                            + str(n_new)
                            + "\n"
                        )
                        break


print("----Zs----")
with Path.open("results/gross_ext_Z_merges.txt", "w") as f:
    f.write("Z merges:\n")
    for i in range(k):
        for j in range(i + 1):
            indices1 = vector_to_indices(Z_logicals[i])
            indices2 = vector_to_indices(Z_logicals[j])

            if (i < 6 and j < 6) or (i > 5 and j > 5):
                print(str(i) + "," + str(j))
                r = 0
                while True:
                    r += 1
                    print("r = ", r)
                    merge_data = external_merge_by_indices(
                        bicycle_code, bicycle_code, indices1, indices2, "Z", r, True
                    )
                    sd = subsystem_distance_GAP(
                        merge_data.Code,
                        merge_data.NewZLogicals,
                        merge_data.NewXLogicals,
                        None,
                    )
                    if sd >= initial_d:
                        print("sd >= initial_d")
                        n_new = len(merge_data.NewQubits)
                        frac = n_new / (2 * n)
                        print(frac)
                        print("num qubits = ", num_data_qubits(merge_data.Code))
                        omega = max(
                            max_measurement_support(merge_data.Code),
                            max_measurement_weight(merge_data.Code),
                        )
                        print("omega = ", omega)
                        f.write(
                            str(i)
                            + ","
                            + str(j)
                            + ","
                            + str(r)
                            + ","
                            + str(frac)
                            + ","
                            + str(omega)
                            + ","
                            + str(n_new)
                            + "\n"
                        )
                        break
