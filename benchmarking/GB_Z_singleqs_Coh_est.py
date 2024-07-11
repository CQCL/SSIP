from ssip.auto_surgeries import (
    generate_classical_line_code,
)
from ssip.basic_functions import (
    find_Z_basis,
    num_data_qubits,
    vector_to_indices,
)
from ssip.lifted_product import generalised_bicycle_code, tensor_product
from ssip.monic_span_checker import is_irreducible_logical, restricted_matrix

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

for j in range(5):
    print("j = ", j)
    l = list_l[j]
    powers_A = list_powers_A[j]
    powers_B = list_powers_B[j]

    gb_code = generalised_bicycle_code(l, powers_A, powers_B)
    initial_d = list_d[j]
    print("GB code, Coh")
    print(num_data_qubits(gb_code))
    hom_basis = find_Z_basis(gb_code)
    num_new_qbs = 0

    num_logicals = min(len(hom_basis), 7)
    for i, v in enumerate(hom_basis):
        if i > 6:
            continue
        ind = vector_to_indices(v)
        if is_irreducible_logical(ind, gb_code.PX):
            r = initial_d
            V = restricted_matrix(ind, gb_code.PX)[0]
            P = generate_classical_line_code(r)
            tens = tensor_product(P, V)

            new_qbs = num_data_qubits(tens) - len(ind)
            num_new_qbs += new_qbs

    print("total num_new_qbs = ", num_new_qbs)
    print(num_new_qbs / num_logicals)

    ####
    print("Surface code")
    surface_code_n = (initial_d) ** 2 + (initial_d - 1) ** 2
    k = len(hom_basis)
    n_init = surface_code_n * k
    print(n_init)
