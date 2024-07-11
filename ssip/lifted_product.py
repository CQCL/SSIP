# Functions to generate tensor & lifted product CSS codes, following https://arxiv.org/abs/2012.04068
import numpy as np

from ssip.basic_functions import BinMatrix
from ssip.code_examples import CSScode


def tensor_product(A: BinMatrix, B: BinMatrix) -> CSScode:
    """Calculate the tensor product in the category of chain complexes
    given two length 1 chain complexes, i.e. classical codes. This produces
    a lengeth 2 chain complex, i.e. CSS code.

    Args:
        A: The first classical code.
        B: The second classical code.

    Return:
        The tensor product qubit CSS code.
    """
    del2a = np.kron(A, np.eye(B.shape[1]))
    del2b = np.kron(np.eye(A.shape[1]), B)
    del2 = np.vstack((del2a, del2b))

    del1a = np.kron(np.eye(A.shape[0]), B)
    del1b = np.kron(A, np.eye(B.shape[0]))
    del1 = np.hstack((del1a, del1b))

    return CSScode(del2.T, del1)


def lift_connected_surface_codes(l: int, L: int) -> CSScode:
    """Construct the lift-connected surface code with base code
    paramaterised by l and lift matrix parameterised by L. See
    p6, https://arxiv.org/abs/2401.02911. Assumes L > 1.

    Args:
        l: Parameter for base code.
        L: Parameter for lift matrix.

    Return:
        The lift-connected surface code LCS(l, L).
    """

    H_rep = np.zeros((l, l + 1))
    H_int = np.zeros((l, l + 1))
    for i in np.arange(l):
        H_rep[i][i] = 1
        H_rep[i][i + 1] = 1
        H_int[i][i + 1] = 1

    identity_L = np.eye(L)
    permutation = np.roll(identity_L, 1, axis=1)

    identity_l = np.eye(l)
    identity_l1 = np.eye(l + 1)

    P_X_rep = np.kron(
        np.hstack((np.kron(identity_l1, H_rep), np.kron(H_rep.T, identity_l))),
        identity_L,
    )
    P_X_int = np.hstack(
        (
            np.kron(np.kron(identity_l1, H_int), permutation),
            np.kron(np.kron(H_int.T, identity_l), permutation.T),
        )
    )
    P_X = P_X_rep + P_X_int

    P_Z_rep = np.kron(
        np.hstack((np.kron(H_rep, identity_l1), np.kron(identity_l, H_rep.T))),
        identity_L,
    )
    P_Z_int = np.hstack(
        (
            np.kron(np.kron(H_int, identity_l1), permutation),
            np.kron(np.kron(identity_l, H_int.T), permutation.T),
        )
    )
    P_Z = P_Z_rep + P_Z_int

    return CSScode(P_Z, P_X)


def bivariate_bicycle_code(
    l: int, m: int, powers_A: tuple[list[int]], powers_B: tuple[list[int]]
) -> CSScode:
    """Construct the bivariate bicycle code with circulants of size l and m
    respectively. See p6, https://arxiv.org/abs/2401.02911.

    Args:
        l: Size of first circulants in the tensor product.
        m: Size of second circulants in the tensor product.
        powers_A: Powers of x and y representing first circulant matrix A.
        powers_B: Powers of x and y representing second circulant matrix B.

    Return:
        The bivariate bicycle code.
    """

    identity_l = np.eye(l)
    permutation_l = np.roll(identity_l, 1, axis=1)
    identity_m = np.eye(m)
    permutation_m = np.roll(identity_m, 1, axis=1)
    x = np.kron(permutation_l, identity_m)
    y = np.kron(identity_l, permutation_m)

    A = np.zeros((x.shape[0], x.shape[1]))
    for i in powers_A[0]:
        A += np.linalg.matrix_power(x, i)
    for i in powers_A[1]:
        A += np.linalg.matrix_power(y, i)

    B = np.zeros((x.shape[0], x.shape[1]))
    for i in powers_B[0]:
        B += np.linalg.matrix_power(x, i)
    for i in powers_B[1]:
        B += np.linalg.matrix_power(y, i)

    P_Z = np.hstack((B.T, A.T))
    P_X = np.hstack((A, B))
    return CSScode(P_Z, P_X)


def generalised_bicycle_code(
    l: int, powers_A: list[int], powers_B: list[int]
) -> CSScode:
    """Construct the generalised bicycle code with circulant of size l,
    see eg https://arxiv.org/abs/1904.02703, https://arxiv.org/abs/1212.6703.

    Args:
        l: Size of circulant.
        powers_A: Powers of x representing first circulant matrix A.
        powers_B: Powers of xrepresenting second circulant matrix B.

    Return:
        The bivariate bicycle code.
    """
    identity_l = np.eye(l)

    A = np.zeros((l, l))
    for i in powers_A:
        A += np.roll(identity_l, i, axis=1)
    B = np.zeros((l, l))
    for i in powers_B:
        B += np.roll(identity_l, i, axis=1)

    P_Z = np.hstack((B.T, A.T))
    P_X = np.hstack((A, B))
    return CSScode(P_Z, P_X)


### Functions specific to bivariate bicycle codes ###


def polynomial_to_qubits(
    poly: list[tuple[int]], l: int, m: int, primed: bool = False
) -> list[int]:
    """Converts a polynomial in two variables into a set
    of qubits in a bivariate bicycle code.

    Args:
        poly: the polynomial in x and y to be converted.
        l: Size of first circulants in the tensor product.
        m: Size of second circulants in the tensor product.
        primed: Whether to index into the primed or
        unprimed block.

    Return:
        The set of indices i.e. qubits in the code.
    """
    indices = []
    for term in poly:
        index = term[0] * m + term[1]
        if primed:
            index += l * m
        indices.append(index)
    return indices


# See https://arxiv.org/abs/2308.07915 p23.
# alpha is a monomial in F_2[x, y], i.e. a pair of integers being powers of (x, y).
# f is a polynomial in F_2[x, y], i.e.
# a list of pairs of integers being powers of (x, y)
def unprimed_X_logical(
    alpha: tuple[int], f: list[tuple[int]], l: int, m: int
) -> list[int]:
    """Picks out a particular X logical in the unprimed block
    of a bivariate bicycle code, see p23 of
    https://arxiv.org/abs/2308.07915.

    Args:
        alpha: A monomial in F_2[x, y], i.e. a pair of integers
        being powers of x, y.
        f: A polynomial in F_2[x, y].
        l: Size of first circulants in the tensor product.
        m: Size of second circulants in the tensor product.

    Return:
        The set of indices i.e. qubits in the code, which the
        X logical has support on.
    """
    new_poly = []
    for mono in f:
        x_power = (alpha[0] + mono[0]) % l
        y_power = (alpha[1] + mono[1]) % m
        new_poly.append((x_power, y_power))
    return polynomial_to_qubits(new_poly, l, m, False)


def primed_X_logical(
    alpha: tuple[int], g: list[tuple[int]], h: list[tuple[int]], l: int, m: int
) -> list[int]:
    """Picks out a particular X logical in the primed block
    of a bivariate bicycle code, see p23 of
    https://arxiv.org/abs/2308.07915.

    Args:
        alpha: A monomial in F_2[x, y], i.e. a pair of integers
        being powers of x, y.
        g: A polynomial in F_2[x, y].
        h: A polynomial in F_2[x, y].
        l: Size of first circulants in the tensor product.
        m: Size of second circulants in the tensor product.

    Return:
        The set of indices i.e. qubits in the code, which the
        X logical has support on.
    """
    new_poly = []
    for mono in g:
        x_power = (alpha[0] + mono[0]) % l
        y_power = (alpha[1] + mono[1]) % m
        new_poly.append((x_power, y_power))
    indices1 = polynomial_to_qubits(new_poly, l, m, False)

    new_poly = []
    for mono in h:
        x_power = (alpha[0] + mono[0]) % l
        y_power = (alpha[1] + mono[1]) % m
        new_poly.append((x_power, y_power))
    indices2 = polynomial_to_qubits(new_poly, l, m, True)

    return indices1 + indices2


def unprimed_Z_logical(
    alpha: tuple[int], h: list[tuple[int]], g: list[tuple[int]], l: int, m: int
) -> list[int]:
    """Picks out a particular Z logical in the unprimed block
    of a bivariate bicycle code, see p23 of
    https://arxiv.org/abs/2308.07915.

    Args:
        alpha: A monomial in F_2[x, y], i.e. a pair of integers
        being powers of x, y.
        h: A polynomial in F_2[x, y].
        g: A polynomial in F_2[x, y].
        l: Size of first circulants in the tensor product.
        m: Size of second circulants in the tensor product.

    Return:
        The set of indices i.e. qubits in the code, which the
        Z logical has support on.
    """
    new_poly = []
    for mono in h:
        x_power = (alpha[0] - mono[0]) % l
        y_power = (alpha[1] - mono[1]) % m
        new_poly.append((x_power, y_power))
    indices1 = polynomial_to_qubits(new_poly, l, m, False)

    new_poly = []
    for mono in g:
        x_power = (alpha[0] - mono[0]) % l
        y_power = (alpha[1] - mono[1]) % m
        new_poly.append((x_power, y_power))
    indices2 = polynomial_to_qubits(new_poly, l, m, True)

    return indices1 + indices2


def primed_Z_logical(
    alpha: tuple[int], f: list[tuple[int]], l: int, m: int
) -> list[int]:
    """Picks out a particular Z logical in the primed block
    of a bivariate bicycle code, see p23 of
    https://arxiv.org/abs/2308.07915.

    Args:
        alpha: A monomial in F_2[x, y], i.e. a pair of integers
        being powers of x, y.
        f: A polynomial in F_2[x, y].
        l: Size of first circulants in the tensor product.
        m: Size of second circulants in the tensor product.

    Return:
        The set of indices i.e. qubits in the code, which the
        Z logical has support on.
    """
    new_poly = []
    for mono in f:
        x_power = (alpha[0] - mono[0]) % l
        y_power = (alpha[1] - mono[1]) % m
        new_poly.append((x_power, y_power))
    return polynomial_to_qubits(new_poly, l, m, True)
