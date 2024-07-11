from pathlib import Path

from ssip.auto_surgeries import (
    MeasureSequence,
    parallel_single_logical_qubit_measurements,
)
from ssip.basic_functions import (
    find_paired_basis,
    find_X_basis,
    find_Z_basis,
    is_valid_CSScode,
    max_measurement_support,
    max_measurement_weight,
    num_data_qubits,
    vector_to_indices,
)
from ssip.distance import subsystem_distance_GAP
from ssip.lifted_product import generalised_bicycle_code
from ssip.monic_span_checker import is_irreducible_logical

list_l = [23, 24, 63, 90, 127]

list_powers_A = [
    [0, 5, 8, 12],
    [0, 2, 8, 15],
    [0, 1, 14, 16, 22],
    [0, 28, 80, 89],
    [0, 15, 20, 28, 66],
]

list_powers_B = [
    [0, 1, 5, 7],
    [0, 2, 12, 17],
    [0, 3, 13, 20, 42],
    [0, 2, 21, 25],
    [0, 58, 59, 100, 121],
]

list_d = [9, 8, 8, 18, 20]

print("----Zs----")
with Path.open("results/gb_par_Z_singleqs.txt", "w") as f:
    f.write("Z merges:\n")
    for j in range(4):
        print("j = ", j)
        l = list_l[j]
        powers_A = list_powers_A[j]
        powers_B = list_powers_B[j]

        gb_code = generalised_bicycle_code(l, powers_A, powers_B)
        initial_n = num_data_qubits(gb_code)
        initial_d = list_d[j]
        print("initial d = ", initial_d)
        hom_basis = find_Z_basis(gb_code)

        index_list = []
        num_logicals = min(int(len(hom_basis) / 2), 7)
        print("num_logicals = ", num_logicals)
        for i, v in enumerate(hom_basis):
            if i == num_logicals:
                break
            ind = vector_to_indices(v)
            if is_irreducible_logical(ind, gb_code.PX):
                index_list.append(ind)

        r = 0
        while True:
            r += 1
            print("r = ", r)

            depths = [r] * len(index_list)
            measure_sequence = MeasureSequence(index_list, depths)
            pushout_data = parallel_single_logical_qubit_measurements(
                gb_code, measure_sequence, "Z", True
            )

            assert is_valid_CSScode(pushout_data.Code)
            print("calculating distance")
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

                n_new = num_data_qubits(pushout_data.Code) - num_data_qubits(gb_code)
                frac = n_new / (num_data_qubits(gb_code))
                print(frac)
                print("num qubits = ", num_data_qubits(pushout_data.Code))
                omega = max(
                    max_measurement_support(pushout_data.Code),
                    max_measurement_weight(pushout_data.Code),
                )
                print("omega = ", omega)
                f.write(
                    str(l)
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


print("----Xs----")
with Path.open("results/gb_par_X_singleqs.txt", "w") as f:
    f.write("X merges:\n")
    for j in range(4):
        print("j = ", j)
        l = list_l[j]
        powers_A = list_powers_A[j]
        powers_B = list_powers_B[j]

        gb_code = generalised_bicycle_code(l, powers_A, powers_B)
        initial_d = list_d[j]
        print("initial d = ", initial_d)
        hom_basis = find_Z_basis(gb_code)
        cohom_basis = find_paired_basis(hom_basis, find_X_basis(gb_code))

        index_list = []
        num_logicals = min(int(len(hom_basis) / 2), 7)
        print("num_logicals = ", num_logicals)
        for i, v in enumerate(cohom_basis):
            if i == num_logicals:
                break
            ind = vector_to_indices(v)
            if is_irreducible_logical(ind, gb_code.PZ):
                index_list.append(ind)

        r = 0
        while True:
            r += 1
            print("r = ", r)

            depths = [r] * len(index_list)
            measure_sequence = MeasureSequence(index_list, depths)
            pushout_data = parallel_single_logical_qubit_measurements(
                gb_code, measure_sequence, "X", True
            )

            assert is_valid_CSScode(pushout_data.Code)
            print("calculating distance")
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

                n_new = num_data_qubits(pushout_data.Code) - num_data_qubits(gb_code)
                frac = n_new / (num_data_qubits(gb_code))
                print(frac)
                print("num qubits = ", num_data_qubits(pushout_data.Code))
                omega = max(
                    max_measurement_support(pushout_data.Code),
                    max_measurement_weight(pushout_data.Code),
                )
                print("omega = ", omega)
                f.write(
                    str(l)
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
