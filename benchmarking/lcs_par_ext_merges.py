from pathlib import Path

import numpy as np

from ssip.auto_surgeries import MergeSequence, parallel_external_merges
from ssip.basic_functions import (
    compose_to_zero,
    direct_sum_codes,
    find_paired_basis,
    find_X_basis,
    find_Z_basis,
    image_basis_calc,
    is_valid_CSScode,
    kernel_basis_calc,
    max_measurement_support,
    max_measurement_weight,
    multiply_F2,
    num_data_qubits,
    vector_to_indices,
)
from ssip.distance import distance_lower_bound_z3
from ssip.lifted_product import lift_connected_surface_codes
from ssip.monic_span_checker import is_irreducible_logical

print("----Zs----")
with Path.open("results/lcs_par_ext_Z_merges.txt", "w") as f:
    f.write("Z merges:\n")
    for l in range(1, 4):
        for L in range(l + 2, l + 5):
            if L > 6:
                continue
            print("l = ", l, "; L = ", L)
            lcs_code = lift_connected_surface_codes(l, L)
            initial_d = min(L, 2 * l + 1)
            print("initial d = ", initial_d)
            initial_n = 2 * num_data_qubits(lcs_code)
            hom_basis = find_Z_basis(lcs_code)
            before_code = direct_sum_codes(lcs_code, lcs_code)

            index_sequence = []
            for v in hom_basis:
                ind = vector_to_indices(v)
                if is_irreducible_logical(ind, lcs_code.PX):
                    index_sequence.append([ind, ind])
                else:
                    print("Not separable")
            r = 0
            while True:
                r += 1
                print("r = ", r)
                depths = [r] * len(index_sequence)
                merge_sequence = MergeSequence(index_sequence, depths)
                merge_data = parallel_external_merges(
                    lcs_code, lcs_code, merge_sequence, "Z", True
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
with Path.open("results/lcs_par_ext_X_merges.txt", "w") as f:
    f.write("X merges:\n")
    for l in range(1, 4):
        for L in range(l + 2, l + 5):
            if L > 6:
                continue
            print("l = ", l, "; L = ", L)
            lcs_code = lift_connected_surface_codes(l, L)
            initial_d = min(L, 2 * l + 1)
            print("initial d = ", initial_d)
            initial_n = 2 * num_data_qubits(lcs_code)
            hom_basis = find_Z_basis(lcs_code)
            cohom_trial_basis = find_X_basis(lcs_code)
            cohom_basis = find_paired_basis(hom_basis, cohom_trial_basis)
            before_code = direct_sum_codes(lcs_code, lcs_code)

            index_sequence = []
            for v in cohom_basis:
                ind = vector_to_indices(v)
                if is_irreducible_logical(ind, lcs_code.PZ):
                    index_sequence.append([ind, ind])
                else:
                    print("Not separable")

            r = 0
            while True:
                r += 1
                print("r = ", r)
                depths = [r] * len(index_sequence)
                merge_sequence = MergeSequence(index_sequence, depths)
                merge_data = parallel_external_merges(
                    lcs_code, lcs_code, merge_sequence, "X", True
                )

                assert is_valid_CSScode(merge_data.Code)
                ker_before = np.array(kernel_basis_calc(before_code.PZ)).T
                ker_after = multiply_F2(merge_data.MergeMap, ker_before)
                assert compose_to_zero(merge_data.Code.PZ, ker_after)

                im_before = np.array(image_basis_calc(before_code.PX.T)).T
                im_after = multiply_F2(merge_data.MergeMap, im_before)
                im_quotient = np.array(
                    kernel_basis_calc(np.array(image_basis_calc(merge_data.Code.PX.T)))
                )
                assert compose_to_zero(im_quotient, im_after)
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
