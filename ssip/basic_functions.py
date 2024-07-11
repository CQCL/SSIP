# Functions for performing elementary calculations with CSS codes,
# relying on the CSS code-homology correspondence.

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

BinMatrix = NDArray[np.uint8]


@dataclass
class CSScode:
    """A qubit Calderbank-Shor-Steane code with a set of stabiliser generators
    is a pair of parity-check matrices over the field F_2.

    Attributes:
        PZ: Parity-check matrix over F_2 for Z generators.
        PX: Parity-check matrix over F_2 for X generators.
    """

    PZ: BinMatrix
    PX: BinMatrix


def flip_code(C: CSScode) -> CSScode:
    """Takes a CSS code and exchanges PX and PZ, giving a code with the same
    parameters but flipped Z and X generators.

    Args:
        C: The CSS code.

    Returns:
        A new CSS code with flipped X and Z generators.
    """
    return CSScode(C.PX, C.PZ)


# add a tuple of vectors over F_2 together
def vec_addition(vecs: Sequence[Sequence[int]]) -> list[int]:
    sum_vec = list(vecs[0])
    for j in range(1, len(vecs)):
        for i in range(len(sum_vec)):
            sum_vec[i] = int(sum_vec[i] != vecs[j][i])
    return sum_vec


def vector_to_indices(v: Sequence[int]) -> Sequence[int]:
    """Converts a vector over F_2 to a list of entries that vector
    has support on.

    Args:
        v: The vector to convert.

    Return:
        The list of indices that vector has non-zero support on.
    """
    indices = [i for i in range(len(v)) if v[i]]
    return indices


def indices_to_vector(indices: Iterable[int], length: int) -> list[int]:
    """Converts a list of indices to a vector with support on those
    indices.

    Args:
        indices: The list of indices.
        length: The desired length of the vector.

    Return:
        The vector with support on those indices.
    """
    v = [0] * length
    for i in indices:
        v[i] = 1
    return v


def dot_product(vec1: BinMatrix, vec2: BinMatrix) -> int:
    """Takes the bilinear symmetric nondegenerate form, i.e. dot product,
    F_2^n x F_2^n -> F_2.

    Args:
        vec1: The first F_2 vector.
        vec2: The second F_2 vector.

    Return:
        The dot product.
    """
    if not len(vec1) == len(vec2):
        raise ValueError("Incompatible lengths of vectors supplied to dot product")
    prod = 0
    for i in range(len(vec1)):
        prod += vec1[i] and vec2[i]
    return prod % 2


def multiply_F2(A: BinMatrix, B: BinMatrix) -> BinMatrix:
    """Multiplies two matrices over F_2.

    Args:
        A: First matrix.
        B: Second matrix.

    Return:
        The matrix AB.
    """
    mult = np.matmul(A, B)
    for i in range(mult.shape[0]):
        for j in range(mult.shape[1]):
            mult[i][j] %= 2
    return mult


def compose_to_zero(A: BinMatrix, B: BinMatrix) -> bool:
    """Check whether two matrices multiply to give zero.

    Args:
        A: First matrix.
        B: Second matrix.

    Return:
        True if AB = 0, False otherwise.
    """
    mult = np.matmul(A, B)
    for row in mult:
        for i in row:
            if int(i) % 2:
                return False
    return True


def is_valid_CSScode(C: CSScode) -> bool:
    """Check whether a CSScode is valid, i.e. the generators commute.

    Args:
        C: The CSScode to check.

    Return:
        True if the generators commute, False otherwise.
    """
    return compose_to_zero(C.PX, C.PZ.T)


def max_measurement_weight(C: CSScode):
    """Finds the maximum weight of any generator in a CSScode.

    Args:
        C: The CSScode to check.

    Return:
        The maximum number of qubits in the support of any generator.
    """

    wX = max(np.sum(C.PX, axis=1))
    wZ = max(np.sum(C.PZ, axis=1))
    return max(wX, wZ)


def max_measurement_support(C: CSScode) -> int:
    """Finds the maximum number of generators any single qubit in a
    CSScode is in the support of.

    Args:
        C: The CSScode to check.

    Return:
        The maximum number of generators any single qubit is in the support of.
    """
    sX = max(np.sum(C.PX, axis=0))
    sZ = max(np.sum(C.PZ, axis=0))
    return max(sX, sZ)


def num_data_qubits(C: CSScode) -> int:
    """Finds the number of data qubits in a CSScode.

    Args:
        C: The CSScode.

    Return:
        The number of data qubits.
    """
    return C.PZ.shape[1]


def num_X_stabs(C: CSScode) -> int:
    """Finds the number of X-type stabiliser generators in a CSScode.

    Args:
        C: The CSScode.

    Return:
        The number of X-type stabiliser generators.
    """
    return C.PX.shape[0]


def num_Z_stabs(C: CSScode) -> int:
    """Finds the number of Z-type stabiliser generators in a CSScode.

    Args:
        C: The CSScode.

    Return:
        The number of Z-type stabiliser generators.
    """
    return C.PZ.shape[0]


# constructs the augmented matrix and applies Gaussian elimination to acquire reduced
# column echelon form
def column_echelon_form(A: BinMatrix, augmented: bool = True):
    def swap_cols(mat: BinMatrix, col1: int, col2: int) -> None:
        if col1 != col2:
            for i in np.arange(mat.shape[0]):
                temp = mat[i][col1]
                mat[i][col1] = mat[i][col2]
                mat[i][col2] = temp

    # add column 1 to column 2, stored in column 2
    def col_addition(mat, col1, col2):
        for i in np.arange(mat.shape[0]):
            mat[i][col2] = mat[i][col1] != mat[i][col2]

    # make augmented matrix
    B = np.array(0)
    if augmented:
        B = np.zeros((A.shape[0] + A.shape[1], A.shape[1]))
        for i in np.arange(A.shape[1]):
            B[A.shape[0] + i][i] = 1
    else:
        B = np.zeros((A.shape[0], A.shape[1]))

    for i in np.arange(A.shape[0]):
        for j in np.arange(A.shape[1]):
            B[i][j] = A[i][j]
    # Gaussian elimination
    h = 0
    k = 0
    while h < B.shape[1] and k < A.shape[0]:
        pivot_exists = False
        pivot_col = 0
        for i in np.arange(h, B.shape[1]):
            if B[k][i]:
                pivot_exists = True
                pivot_col = i
        if not pivot_exists:
            k += 1

        else:
            swap_cols(B, h, pivot_col)
            for i in np.arange(0, B.shape[1]):
                if B[k][i] and i != h:
                    col_addition(B, h, i)
            k += 1
            h += 1
    return B


def kernel_basis_calc(A: BinMatrix) -> list[list[int]]:
    """Calculates a basis of the kernel of a matrix.

    Args:
        A: The matrix.

    Return:
        A basis of the kernel of the matrix.
    """
    B = column_echelon_form(A, True)
    basis = []
    for j in np.arange(B.shape[1] - 1, -1, -1):
        all_zeros = True
        for i in np.arange(A.shape[0]):
            if B[i][j]:
                all_zeros = False
                break
        if not all_zeros:
            break
        basis_vec = [0] * (B.shape[1])
        for i in np.arange(A.shape[0], B.shape[0]):
            basis_vec[i - A.shape[0]] = B[i][j]
        basis.append(basis_vec)

    return basis


def kernel_all_vectors(A: BinMatrix) -> list[list[int]]:
    """Calculates all nonzero vectors in the kernel of a matrix.

    Args:
        A: The matrix.

    Return:
        All nonzero vectors in the kernel of A.
    """
    basis = kernel_basis_calc(A)
    all_combs = []
    for i in range(1, len(basis) + 1):
        all_combs += list(combinations(basis, i))
    for i in range(len(all_combs)):
        all_combs[i] = vec_addition(all_combs[i])

    return all_combs


def image_basis_calc(A: BinMatrix) -> list[list[int]]:
    """Calculates a basis of the image of a matrix.

    Args:
        A: The matrix.

    Return:
        A basis of the image of the matrix.
    """
    B = column_echelon_form(A, False)
    basis = []
    for j in np.arange(B.shape[1]):
        all_zeros = True
        for i in np.arange(B.shape[0]):
            if B[i][j]:
                all_zeros = False
        if all_zeros:
            break
        else:
            basis_vec = [0] * (B.shape[0])
            for i in np.arange(B.shape[0]):
                basis_vec[i] = B[i][j]
            basis.append(basis_vec)

    return basis


def image_all_vectors(A: BinMatrix) -> list[list[int]]:
    """Calculates all nonzero vectors in the image of a matrix.

    Args:
        A: The matrix.

    Return:
        All nonzero vectors in the image of A.
    """
    basis = image_basis_calc(A)
    all_combs = []
    for i in range(1, len(basis) + 1):
        all_combs += list(combinations(basis, i))
    for i in range(len(all_combs)):
        all_combs[i] = vec_addition(all_combs[i])

    return all_combs


def num_logical_qubits(C: CSScode) -> int:
    """Calculate the number of logical qubits of a CSScode.

    Args:
        C: The CSScode.

    Return:
        The number of logical qubits of C.
    """
    im = image_basis_calc(C.PZ.T)
    ker = kernel_basis_calc(C.PX)
    return len(ker) - len(im)


# Finds a basis for the homology space, i.e. the quotient ker(B)/im(A).
# Works by first finding the quotient morphism Q: F_2^n -> F_2^n/im(A),
# then performing Gaussian elimination to calculate the linearly independent
# vectors in ker(B) under the equivalence /imA.
def find_homology_basis(A: BinMatrix, B: BinMatrix) -> list[list[int]]:
    im_basis = np.array(image_basis_calc(A))
    im_dim = im_basis.shape[0]
    quotient_map = np.array(kernel_basis_calc(im_basis))

    ker_basis = np.array(kernel_basis_calc(B)).T
    ker_dim = ker_basis.shape[1]

    hom_dim = ker_dim - im_dim
    if hom_dim < 0:
        raise ValueError("Kernel is smaller than image")

    quotiented_vecs = multiply_F2(quotient_map, ker_basis)
    height = quotiented_vecs.shape[0]
    mat = column_echelon_form(quotiented_vecs, True)

    vecs_to_add = [0] * hom_dim
    for i in np.arange(hom_dim):
        v = [0] * (mat.shape[0] - height)
        for j in np.arange(height, mat.shape[0]):
            v[j - height] = mat[j][i]
        vecs_to_add[i] = v

    hom_basis = [0] * hom_dim
    for i in np.arange(hom_dim):
        vecs = [
            list(ker_basis[:, j])
            for j in np.arange(len(vecs_to_add[i]))
            if vecs_to_add[i][j]
        ]

        hom_basis[i] = vec_addition(vecs)

    return hom_basis


def find_Z_basis(C: CSScode) -> list[list[int]]:
    """Calculate a basis of Z logical operators of a CSScode.

    Args:
        C: The CSScode.

    Return:
        A basis of Z logical operators of C.
    """
    return find_homology_basis(C.PZ.T, C.PX)


def find_X_basis(C: CSScode) -> list[list[int]]:
    """Calculate a basis of X logical operators of a CSScode.

    Args:
        C: The CSScode.

    Return:
        A basis of X logical operators of C.
    """
    return find_homology_basis(C.PX.T, C.PZ)


def find_paired_basis(
    basis1: list[list[int]], basis2: list[list[int]]
) -> list[list[int]]:
    """Calculate a paired basis of logical operators of a CSScode. That is,
    given two bases of the homology H_1 and cohomology H^1 respectively,
    find and return a basis of H^1 which is 'paired' with that of H_1 by the
    nondegenerate bilinear form, i.e. the dot product.

    Args:
        basis1: The basis of H_1.
        basis2: The basis of H^1.

    Return:
        A basis of H^1 which is paired with basis1.
    """
    if len(basis1) != len(basis2):
        raise ValueError(
            "Bases have lengths " + str(len(basis1)) + " & " + str(len(basis2))
        )
    m = len(basis1)

    mat = np.zeros((m, m), dtype=np.uint8)
    for i in range(m):
        for j in range(m):
            mat[i][j] = dot_product(basis1[i], basis2[j])
    B = column_echelon_form(mat, True)
    replacement_basis = []
    for j in np.arange(B.shape[1]):
        vecs_to_add = [
            basis2[i - B.shape[1]]
            for i in np.arange(mat.shape[0], B.shape[0])
            if B[i][j]
        ]

        sum_vec = vec_addition(vecs_to_add)
        replacement_basis.append(sum_vec)

    return replacement_basis


# given a (co)chain complex C2 -> C1 -> C0, with A: C2 -> C1 and B: C1 -> C0
# returns the systolic distance at degree 1
def systolic_distance(A: BinMatrix, B: BinMatrix) -> int:
    quotient_map = np.array(kernel_basis_calc(A.T))
    ker_vecs = kernel_all_vectors(B)
    sys_distance = len(ker_vecs[0])
    for v in ker_vecs:
        weight = int(sum(v))
        if weight < sys_distance and not compose_to_zero(quotient_map, np.array([v]).T):
            sys_distance = weight
    return sys_distance


def direct_sum_matrices(A: BinMatrix, B: BinMatrix) -> BinMatrix:
    """Compute the direct sum of two matrices.

    Args:
        A: The top left matrix.
        B: The bottom right matrix.

    Return:
        The direct sum of A and B.
    """
    rows = A.shape[0] + B.shape[0]
    cols = A.shape[1] + B.shape[1]
    direct_sumAB = np.zeros((rows, cols), dtype=np.uint8)
    direct_sumAB[: A.shape[0], : A.shape[1]] = A
    direct_sumAB[A.shape[0] :, A.shape[1] :] = B
    return direct_sumAB


def direct_sum_codes(C: CSScode, D: CSScode) -> CSScode:
    """Compute the direct sum of two codes (as chain or cochain complexes).

    Args:
        C: The first code.
        D: The second code.

    Return:
        The direct sum of C and D.
    """
    return CSScode(direct_sum_matrices(C.PZ, D.PZ), direct_sum_matrices(C.PX, D.PX))
