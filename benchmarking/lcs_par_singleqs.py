from pathlib import Path

import numpy as np

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
from ssip.distance import distance_lower_bound_z3
from ssip.lifted_product import lift_connected_surface_codes
from ssip.monic_span_checker import is_irreducible_logical

print("----Zs----")
with Path.open("results/lcs_par_Z_singleqs.txt", "w") as f:
    f.write("Z merges:\n")
    for l in range(1, 4):
        for L in range(l + 2, l + 5):
            if L > 6:
                continue
            print("l = ", l, "; L = ", L)
            lcs_code = lift_connected_surface_codes(l, L)
            initial_d = min(L, 2 * l + 1)
            print("initial d = ", initial_d)
            hom_basis = find_Z_basis(lcs_code)

            index_list = []
            num_logicals = int(len(hom_basis) / 2)
            print("num_logicals = ", num_logicals)
            for i, v in enumerate(hom_basis):
                if i == num_logicals:
                    break
                ind = vector_to_indices(v)
                if is_irreducible_logical(ind, lcs_code.PX):
                    index_list.append(ind)

            r = 0
            while True:
                r += 1
                print("r = ", r)

                depths = [r] * len(index_list)
                measure_sequence = MeasureSequence(index_list, depths)
                pushout_data = parallel_single_logical_qubit_measurements(
                    lcs_code, measure_sequence, "Z", True
                )

                assert is_valid_CSScode(pushout_data.Code)
                lower1 = distance_lower_bound_z3(
                    pushout_data.Code.PX,
                    np.array(pushout_data.OldXLogicals),
                    initial_d - 1,
                )
                lower2 = distance_lower_bound_z3(
                    pushout_data.Code.PZ,
                    np.array(pushout_data.OldZLogicals),
                    initial_d - 1,
                )
                if lower1 is None and lower2 is None:
                    print("sd >= initial_d")
                    print("r = ", r)

                    n_new = num_data_qubits(pushout_data.Code) - num_data_qubits(
                        lcs_code
                    )
                    frac = n_new / (num_data_qubits(lcs_code))
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
                        + str(L)
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
with Path.open("results/lcs_par_X_singleqs.txt", "w") as f:
    f.write("X merges:\n")
    for l in range(1, 4):
        for L in range(l + 2, l + 5):
            if L > 6:
                continue
            print("l = ", l, "; L = ", L)
            lcs_code = lift_connected_surface_codes(l, L)
            initial_d = min(L, 2 * l + 1)
            print("initial d = ", initial_d)
            hom_basis = find_Z_basis(lcs_code)
            cohom_trial_basis = find_X_basis(lcs_code)
            cohom_basis = find_paired_basis(hom_basis, cohom_trial_basis)

            index_list = []
            num_logicals = int(len(hom_basis) / 2)
            print("num_logicals = ", num_logicals)
            for i, v in enumerate(cohom_basis):
                if i == num_logicals:
                    break
                ind = vector_to_indices(v)
                if is_irreducible_logical(ind, lcs_code.PZ):
                    index_list.append(ind)

            r = 0
            while True:
                r += 1
                print("r = ", r)

                depths = [r] * len(index_list)
                measure_sequence = MeasureSequence(index_list, depths)
                pushout_data = parallel_single_logical_qubit_measurements(
                    lcs_code, measure_sequence, "X", True
                )

                assert is_valid_CSScode(pushout_data.Code)
                lower1 = distance_lower_bound_z3(
                    pushout_data.Code.PX,
                    np.array(pushout_data.OldXLogicals),
                    initial_d - 1,
                )
                lower2 = distance_lower_bound_z3(
                    pushout_data.Code.PZ,
                    np.array(pushout_data.OldZLogicals),
                    initial_d - 1,
                )
                if lower1 is None and lower2 is None:
                    print("sd >= initial_d")
                    print("r = ", r)

                    n_new = num_data_qubits(pushout_data.Code) - num_data_qubits(
                        lcs_code
                    )
                    frac = n_new / (num_data_qubits(lcs_code))
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
                        + str(L)
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
