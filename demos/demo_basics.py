# A CSScode object in SSIP is just a pair of numpy arrays, with PZ and PX for the Z and X-type parity-check matrices respectively.
from ssip.code_examples import Shor_code

print("PZ = ", Shor_code.PZ)
print("PX = ", Shor_code.PX)

# We can obtain basic properties of this code.
from ssip.basic_functions import (
    find_paired_basis,
    find_X_basis,
    find_Z_basis,
    is_valid_CSScode,
    max_measurement_support,
    max_measurement_weight,
    num_data_qubits,
    num_logical_qubits,
    num_X_stabs,
    num_Z_stabs,
)

# Checks that the stabilisers commute in a CSS code
print(is_valid_CSScode(Shor_code))

# Calculate some relevant properties
n = num_data_qubits(Shor_code)
print("n = ", n)
k = num_logical_qubits(Shor_code)
print("k = ", k)
print(num_Z_stabs(Shor_code))
print(num_X_stabs(Shor_code))

# This is the maximum number of measurements any qubit is in the support of
print(max_measurement_support(Shor_code))
# This is the maximum number of qubits any measurement has in its support
print(max_measurement_weight(Shor_code))

# Compute an arbitrary tensor decomposition of the logical space into qubits, then return representatives of the equivalence class
# of logical Z operators on these qubits. The Shor code only has 1 logical qubit so only 1 representative is returned.
Z_basis = find_Z_basis(Shor_code)
print(Z_basis)

# The same for X logicals
X_basis = find_X_basis(Shor_code)
# We would like the bases to coincide, so that the tensor decomposition is consistent and logicals on the same qubits
# anticommute. We call these `paired bases`. The following function takes two bases, one for Z and X, and returns a
# modified X basis which is consistent with the Z.
new_X_basis = find_paired_basis(Z_basis, X_basis)
print(new_X_basis)

# We can also calculate the distance of codes. There are several ways of doing this, but lets start with a naive one.
from ssip.distance import code_distance

d = code_distance(Shor_code)
print("d = ", d)

# On codes with low distances it is more efficient to use Z3.
from ssip.distance import distance_z3

assert distance_z3(Shor_code) == d

# We can take two parallel codeblocks and put them into the same block.
from ssip.basic_functions import direct_sum_codes

new_code = direct_sum_codes(Shor_code, Shor_code)
assert num_data_qubits(new_code) == 2 * n
assert num_logical_qubits(new_code) == 2 * k
assert distance_z3(new_code) == d
