# To perform surgery between codes we must identify logicals which are irreducible.
from ssip.code_examples import Shor_code

v = [1, 0, 0, 1, 0, 0, 1, 0, 0]
from ssip.basic_functions import vector_to_indices

# It is more efficient to pass around low weight vectors as the indices on which
# they have support
indices = vector_to_indices(v)

from ssip.monic_span_checker import is_irreducible_logical

# This method checks that a Z operator is indeed irreducible, although it does not
# verify that it is a nontrivial logical.
assert is_irreducible_logical(indices, Shor_code.PX)


# Now we can perform a merge between two Shor codeblocks along copies of v.
from ssip.basic_functions import is_valid_CSScode
from ssip.pushouts import external_merge_by_indices

# This is a logical parity Z-measurement.
merged_code = external_merge_by_indices(
    Shor_code, Shor_code, indices, indices, basis="Z"
)
assert is_valid_CSScode(merged_code)

# We can check that this merge has retained the distance of 3
from ssip.distance import distance_z3

assert distance_z3(merged_code) == 3

# This has quotiented the two logicals into the same equivalence class, so...
from ssip.basic_functions import num_logical_qubits

assert num_logical_qubits(merged_code) == 1


# It is also possible to merge between different codes
from ssip.code_examples import Surface3_code

u = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
indices2 = vector_to_indices(u)
assert is_irreducible_logical(indices2, Surface3_code.PX)

merged_code = external_merge_by_indices(
    Shor_code, Surface3_code, indices, indices2, basis="Z"
)
assert is_valid_CSScode(merged_code)
assert num_logical_qubits(merged_code) == 1
assert distance_z3(merged_code) == 3


# Merges can optionally return additional data, captured in a MergeResult object.
merge_result = external_merge_by_indices(
    Shor_code, Surface3_code, indices, indices2, basis="Z", return_data=True
)
# The output code is accessible as part of this MergeResult
merged_code = merge_result.Code
assert is_valid_CSScode(merged_code)
# But we also have access to the new data qubits and stabilisers introduced, among other things.
print(merge_result.NewQubits)
print(merge_result.NewZStabs)
print(merge_result.NewXStabs)
# As you can see, there are 2 new data qubits, 3 new Z stabilisers, and no new X stabilisers.

# Sometimes a merge will not preserve the distance, and so we can increase the `depth` of
# the merge, that is the size of the tensor product code used to bridge between the two codes.
merge_result = external_merge_by_indices(
    Shor_code, Surface3_code, indices, indices2, basis="Z", depth=3, return_data=True
)
# Increasing the depth increases the number of data qubits and stabilisers introduced.
assert is_valid_CSScode(merge_result.Code)
print(merge_result.NewQubits)
print(merge_result.NewZStabs)
print(merge_result.NewXStabs)
