from pathlib import Path

import numpy as np

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
from ssip.pushouts import external_merge_by_indices

print("----Zs----")
with Path.open("results/lcs_ext_Z_merges.txt", "w") as f:
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

            for i, v in enumerate(hom_basis):
                print(" i = ", i)
                ind = vector_to_indices(v)
                if is_irreducible_logical(ind, lcs_code.PX):
                    r = 0
                    while True:
                        r += 1
                        merge_data = external_merge_by_indices(
                            lcs_code, lcs_code, ind, ind, "Z", r, True
                        )

                        assert is_valid_CSScode(merge_data.Code)
                        lower1 = distance_lower_bound_z3(
                            merge_data.Code.PX,
                            np.array(merge_data.OldXLogicals),
                            initial_d - 1,
                        )
                        lower2 = distance_lower_bound_z3(
                            merge_data.Code.PZ,
                            np.array(merge_data.OldZLogicals),
                            initial_d - 1,
                        )
                        if lower1 is None and lower2 is None:
                            print("sd >= initial_d")
                            print("r = ", r)
                            n_new = len(merge_data.NewQubits)
                            frac = n_new / (2 * num_data_qubits(lcs_code))
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
                                + str(L)
                                + ","
                                + str(i)
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
                else:
                    print("Logical not irreducible!")


print("----Xs----")
with Path.open("results/lcs_ext_X_merges.txt", "w") as f:
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

            for i, v in enumerate(cohom_basis):
                print(" i = ", i)
                ind = vector_to_indices(v)
                if is_irreducible_logical(ind, lcs_code.PZ):
                    r = 0
                    while True:
                        r += 1
                        merge_data = external_merge_by_indices(
                            lcs_code, lcs_code, ind, ind, "X", r, True
                        )
                        lower1 = distance_lower_bound_z3(
                            merge_data.Code.PX,
                            np.array(merge_data.OldXLogicals),
                            initial_d - 1,
                        )
                        lower2 = distance_lower_bound_z3(
                            merge_data.Code.PZ,
                            np.array(merge_data.OldZLogicals),
                            initial_d - 1,
                        )
                        if lower1 is None and lower2 is None:
                            print("sd >= initial_d")
                            print("r = ", r)
                            n_new = len(merge_data.NewQubits)
                            frac = n_new / (2 * num_data_qubits(lcs_code))
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
                                + str(L)
                                + ","
                                + str(i)
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
                else:
                    print("Logical not irreducible!")
