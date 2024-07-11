from ssip.auto_surgeries import (
    generate_classical_line_code,
)
from ssip.basic_functions import (
    find_Z_basis,
    num_data_qubits,
    vector_to_indices,
)
from ssip.lifted_product import lift_connected_surface_codes, tensor_product
from ssip.monic_span_checker import is_irreducible_logical, restricted_matrix

for l in range(1, 4):
    for L in range(l + 2, l + 5):
        if L > 6:
            continue
        print("l = ", l, "; L = ", L)
        lcs_code = lift_connected_surface_codes(l, L)
        initial_d = min(L, 2 * l + 1)
        print("LCS code, Coh")
        print(num_data_qubits(lcs_code))
        hom_basis = find_Z_basis(lcs_code)
        num_logicals = int(len(hom_basis) / 2)
        num_new_qbs = 0
        for i, v in enumerate(hom_basis):
            if i == num_logicals:
                break
            ind = vector_to_indices(v)
            if is_irreducible_logical(ind, lcs_code.PX):
                r = initial_d
                V = restricted_matrix(ind, lcs_code.PX)[0]
                P = generate_classical_line_code(r)
                tens = tensor_product(P, V)

                new_qbs = num_data_qubits(tens) - len(ind)
                num_new_qbs += new_qbs

        print("total num_new_qbs = ", num_new_qbs)

        ####
        print("Surface code")
        surface_code_n = (initial_d) ** 2 + (initial_d - 1) ** 2
        k = L
        n_init = surface_code_n * k
        print(n_init)
