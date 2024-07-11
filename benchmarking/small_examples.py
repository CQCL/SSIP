from ssip.basic_functions import (
    dot_product,
    indices_to_vector,
    is_valid_CSScode,
    max_measurement_support,
    max_measurement_weight,
    num_data_qubits,
    num_logical_qubits,
)
from ssip.code_examples import (
    QRM_code,
    Rotated17_code,
    Shor_code,
    Steane_code,
    Surface3_code,
)
from ssip.distance import distance_z3
from ssip.monic_span_checker import is_irreducible_logical
from ssip.pushouts import external_merge_by_indices

list_codes = [Shor_code, QRM_code, Steane_code, Rotated17_code, Surface3_code]
Z_logicals = [[0, 3, 6], [0, 3, 4], [0, 1, 2], [0, 1, 2], [0, 1, 2]]
X_logicals = [[6, 7, 8], [0, 1, 2, 3, 4, 5, 6], [0, 1, 2], [0, 3, 6], [1, 6, 11]]
Ws = []

for i in range(len(list_codes)):
    n = num_data_qubits(list_codes[i])
    Z_vector = indices_to_vector(Z_logicals[i], n)
    X_vector = indices_to_vector(X_logicals[i], n)
    assert dot_product(Z_vector, X_vector) == 1

    assert is_irreducible_logical(Z_logicals[i], list_codes[i].PX)
    assert is_irreducible_logical(X_logicals[i], list_codes[i].PZ)

    Ws.append(
        max(
            max_measurement_weight(list_codes[i]),
            max_measurement_support(list_codes[i]),
        )
    )

print("Zs")
for i in range(len(list_codes)):
    for j in range(i + 1):
        merged_code = external_merge_by_indices(
            list_codes[i], list_codes[j], Z_logicals[i], Z_logicals[j], "Z"
        )
        assert is_valid_CSScode(merged_code)
        print("i = ", i, ", j = ", j)
        print("k = ", num_logical_qubits(merged_code))
        print("d = ", distance_z3(merged_code))
        w_before = max(Ws[i], Ws[j])
        w_after = max(
            max_measurement_weight(merged_code), max_measurement_support(merged_code)
        )
        print("w = ", w_after)
        print(int(w_after - w_before))

print("Xs")
for i in range(len(list_codes)):
    if i == 1:
        continue
    for j in range(i + 1):
        if j == 1:
            continue
        merged_code = external_merge_by_indices(
            list_codes[i], list_codes[j], X_logicals[i], X_logicals[j], "X"
        )
        assert is_valid_CSScode(merged_code)
        print("i = ", i, ", j = ", j)
        print("k = ", num_logical_qubits(merged_code))
        print("d = ", distance_z3(merged_code))
        w_before = max(Ws[i], Ws[j])
        w_after = max(
            max_measurement_weight(merged_code), max_measurement_support(merged_code)
        )
        print("w = ", w_after)
        print(int(w_after - w_before))
