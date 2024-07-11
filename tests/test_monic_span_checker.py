import numpy as np

from ssip.basic_functions import compose_to_zero, vector_to_indices
from ssip.code_examples import (
    QRM_B,
    GTri_B,
    Shor_B,
    Steane_B,
    Surface2_B,
    Surface3_B,
)
from ssip.monic_span_checker import (
    is_irreducible_logical,
    monic_span,
    restricted_matrix,
)


# hack to define equality between boolean numpy arrays
def boolean_matrix_equality(mat1, mat2):
    return not (mat1 - mat2).any()


##### Tests #####


# test when matrices are row/column permutations of each other
def test_permutation():
    mat1 = np.array([[1, 1], [1, 0]])
    mat2 = np.array([[0, 1], [1, 1]])
    mat3 = np.array([[0, 0], [1, 0]])
    mat4 = np.array([[1, 1]])
    mat5 = np.array([[1]])
    assert monic_span(mat1, mat2) is not None  # check that there is a monic span
    assert monic_span(mat1, mat3) is None
    assert monic_span(mat1, mat4) is None
    assert monic_span(mat4, mat5) is None


# monic span between two identical Shor codes
def test_shor_monic_span():
    v = [1, 0, 0, 1, 0, 0, 1, 0, 0]
    indices = [0, 3, 6]
    restr = restricted_matrix(indices, Shor_B)
    assert compose_to_zero(Shor_B, np.array([v]).T)
    assert restr[1] == [0, 1]
    assert boolean_matrix_equality(restr[0], np.array([[1, 1, 0], [1, 0, 1]]))

    assert is_irreducible_logical(indices, Shor_B)

    v2 = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert compose_to_zero(Shor_B, np.array([v2]).T)
    assert not is_irreducible_logical(vector_to_indices(v2), Shor_B)

    assert monic_span(restr[0], restr[0]) is not None


# monic span between Shor code and Surface code
def test_shor_surface_monic_span():
    v1 = [1, 0, 0, 1, 0, 0, 1, 0, 0]
    ind1 = vector_to_indices(v1)
    v2 = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ind2 = vector_to_indices(v2)

    restr1 = restricted_matrix(ind1, Shor_B)[0]
    assert boolean_matrix_equality(restr1, np.array([[1, 1, 0], [1, 0, 1]]))
    restr2 = restricted_matrix(ind2, Surface3_B)[0]
    assert boolean_matrix_equality(restr2, np.array([[1, 1, 0], [0, 1, 1]]))

    assert is_irreducible_logical(ind2, Surface3_B)

    assert monic_span(restr1, restr2) is not None


# T-state injection from QRM code into a surface code
def test_surface_qrm_monic_span():
    logical_to_merge1 = [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    indices1 = vector_to_indices(logical_to_merge1)

    assert is_irreducible_logical(indices1, QRM_B)

    logical_to_merge2 = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    indices2 = vector_to_indices(logical_to_merge2)

    restr1 = restricted_matrix(indices1, QRM_B)[0]
    restr2 = restricted_matrix(indices2, Surface3_B)[0]

    assert monic_span(restr1, restr2) is not None


# T-state injection from QRM code into a Shor code
def test_shor_qrm_monic_span():
    qrm_logical = [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ind1 = vector_to_indices(qrm_logical)

    shor_logical = [1, 0, 0, 1, 0, 0, 1, 0, 0]
    ind2 = vector_to_indices(shor_logical)

    restr1 = restricted_matrix(ind1, QRM_B)[0]
    restr2 = restricted_matrix(ind2, Shor_B)[0]

    assert monic_span(restr1, restr2) is not None


# Colour code to surface code merge, from H. P. Nautrup, N. Friis and H. J. Briegel,
# Fault-tolerant interface between quantum memories and quantum processors,
# Nat. Commun. 8, 1321 (2017)
# One can use this to e.g. inject an S gate into the surface code.
def test_colour_surface_monic_span():
    colour_logical = [1, 1, 1, 0, 0, 0, 0]
    ind1 = vector_to_indices(colour_logical)
    assert is_irreducible_logical(ind1, Steane_B)

    surface_logical = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ind2 = vector_to_indices(surface_logical)
    assert is_irreducible_logical(ind2, Surface3_B)

    restr1 = restricted_matrix(ind1, Steane_B)[0]
    restr2 = restricted_matrix(ind2, Surface3_B)[0]

    assert monic_span(restr1, restr2) is not None


def test_colour_shor_monic_span():
    colour_logical = [1, 1, 1, 0, 0, 0, 0]
    ind1 = vector_to_indices(colour_logical)

    shor_logical = [1, 0, 0, 1, 0, 0, 1, 0, 0]
    ind2 = vector_to_indices(shor_logical)
    restr1 = restricted_matrix(ind1, Steane_B)[0]
    restr2 = restricted_matrix(ind2, Shor_B)[0]

    assert monic_span(restr1, restr2) is not None


def test_gtri_monic_span():
    surface_logical = [1, 1, 0, 0, 0]
    ind1 = vector_to_indices(surface_logical)

    gtri_logical = [1, 1, 0, 0, 0, 0, 0, 0]
    ind2 = vector_to_indices(gtri_logical)

    restr1 = restricted_matrix(ind1, Surface2_B)[0]
    restr2 = restricted_matrix(ind2, GTri_B)[0]

    assert monic_span(restr1, restr2) is not None
