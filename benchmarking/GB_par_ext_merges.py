from pathlib import Path

from ssip.auto_surgeries import MergeSequence, parallel_external_merges
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
from ssip.distance import (
    subsystem_distance_GAP,
)
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
with Path.open("results/gb_par_ext_Z_merges.txt", "w") as f:
    f.write("Z merges:\n")
    for j in range(5):
        print("j = ", j)
        l = list_l[j]
        powers_A = list_powers_A[j]
        powers_B = list_powers_B[j]

        gb_code = generalised_bicycle_code(l, powers_A, powers_B)
        initial_n = 2 * num_data_qubits(gb_code)
        initial_d = list_d[j]
        print("initial d = ", initial_d)
        hom_basis = find_Z_basis(gb_code)

        index_sequence = []
        num_logicals = 0
        for v in hom_basis:
            if num_logicals == 7:
                break
            ind = vector_to_indices(v)
            if is_irreducible_logical(ind, gb_code.PX):
                index_sequence.append([ind, ind])
                num_logicals += 1
            else:
                print("Not separable")

        r = 0
        while True:
            r += 1
            print("r = ", r)
            depths = [r] * len(index_sequence)
            merge_sequence = MergeSequence(index_sequence, depths)
            print(len(merge_sequence.IndexSequence))
            merge_data = parallel_external_merges(
                gb_code, gb_code, merge_sequence, "Z", True
            )
            assert is_valid_CSScode(merge_data.Code)
            print("calculating distance")
            sd = subsystem_distance_GAP(
                merge_data.Code,
                merge_data.NewZLogicals,
                merge_data.NewXLogicals,
                None,
                10000,
            )
            if sd >= initial_d:
                print("sd >= initial_d")
                n_new = len(merge_data.NewQubits)
                frac = n_new / initial_n
                print(frac)
                print("num qubits = ", num_data_qubits(merge_data.Code))
                omega = max(
                    max_measurement_support(merge_data.Code),
                    max_measurement_weight(merge_data.Code),
                )
                print("omega = ", omega)
                f.write(
                    str(l)
                    + ","
                    + str(r)
                    + ","
                    + str(frac)
                    + ","
                    + str(omega)
                    + ","
                    + str(initial_n)
                    + ","
                    + str(n_new)
                    + "\n"
                )
                break


print("----Xs----")
with Path.open("results/gb_par_ext_X_merges.txt", "w") as f:
    f.write("X merges:\n")
    for j in range(5):
        print("j = ", j)
        l = list_l[j]
        powers_A = list_powers_A[j]
        powers_B = list_powers_B[j]

        gb_code = generalised_bicycle_code(l, powers_A, powers_B)
        initial_n = 2 * num_data_qubits(gb_code)
        initial_d = list_d[j]
        print("initial d = ", initial_d)
        hom_basis = find_Z_basis(gb_code)
        cohom_basis = find_paired_basis(hom_basis, find_X_basis(gb_code))

        index_sequence = []
        num_logicals = 0
        for v in cohom_basis:
            if num_logicals == 7:
                break
            ind = vector_to_indices(v)
            if is_irreducible_logical(ind, gb_code.PZ):
                index_sequence.append([ind, ind])
                num_logicals += 1
            else:
                print("Not separable")

        r = 0
        while True:
            r += 1
            print("r = ", r)

            depths = [r] * len(index_sequence)
            merge_sequence = MergeSequence(index_sequence, depths)
            merge_data = parallel_external_merges(
                gb_code, gb_code, merge_sequence, "X", True
            )

            assert is_valid_CSScode(merge_data.Code)
            print("calculating distance")
            sd = subsystem_distance_GAP(
                merge_data.Code,
                merge_data.NewZLogicals,
                merge_data.NewXLogicals,
                None,
                10000,
            )
            if sd >= initial_d:
                print("sd >= initial_d")
                n_new = len(merge_data.NewQubits)
                frac = n_new / initial_n
                print(frac)
                print("num qubits = ", num_data_qubits(merge_data.Code))
                omega = max(
                    max_measurement_support(merge_data.Code),
                    max_measurement_weight(merge_data.Code),
                )
                print("omega = ", omega)
                f.write(
                    str(l)
                    + ","
                    + str(r)
                    + ","
                    + str(frac)
                    + ","
                    + str(omega)
                    + ","
                    + str(initial_n)
                    + ","
                    + str(n_new)
                    + "\n"
                )
                break
