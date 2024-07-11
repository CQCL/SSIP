# Functions for checking that two logical operators belonging to different codes do
# indeed form a basis-preserving irreducible monic span.
from collections.abc import Iterable

import networkx as nx
import numpy as np

from ssip.basic_functions import (
    BinMatrix,
    compose_to_zero,
    indices_to_vector,
    kernel_basis_calc,
)


# given a set of indices indicating the support of a vector in the domain
# calculates the matrix restricted to that support
def restricted_matrix(indices: list[int], B: BinMatrix) -> (BinMatrix, list[int]):
    width = len(indices)
    R = np.zeros((B.shape[0], width))
    # restrict columns based on indices in support
    for i in range(B.shape[0]):
        for j in range(width):
            R[i][j] = B[i][indices[j]]

    # restrict rows by removing all-zero rows
    rows_to_keep = []
    for i in range(R.shape[0]):
        all_zeros = True
        for j in range(width):
            if R[i][j]:
                all_zeros = False
                break
        if not all_zeros:
            rows_to_keep.append(i)

    height = len(rows_to_keep)
    R_2 = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            R_2[i][j] = R[rows_to_keep[i]][j]

    return (R_2, rows_to_keep)


# checks whether a logical contains any other logicals in its support
def is_irreducible_logical(indices: Iterable[int], B: BinMatrix) -> bool:
    """Takes a vector v in ker(B) and checks whether it is irreducible,
    i.e. has no subvectors in its support which are also in B. Does
    not check that the vector is a nontrivial logical, i.e. is not in
    im(A) for some check matrix A.

    Args:
        indices: The list of indices in the support of v.
        B: The check matrix for v.

    Returns:
        True if v is in ker(B) and there is no subvector of v which
        is also in ker(B). False otherwise.
    """
    vec = np.array([indices_to_vector(indices, B.shape[1])]).T
    if not compose_to_zero(B, vec):
        return False
    restr = restricted_matrix(indices, B)[0]
    return len(kernel_basis_calc(restr)) == 1


def monic_span(M: BinMatrix, N: BinMatrix) -> (dict, dict):
    """Checks that there is a monic span with a logical operator subcomplex at the apex,
    with each morphism
    mapping basis elements to basis elements, and returns the relevant data
    required to construct that span (there may be multiple,
    in which case it returns the first it finds).
    We approach this by checking that two boolean matrices M and N
    are equivalent up to permutation
    of rows/columns, i.e. PMQ = N for some permutation matrices P, Q.
    This is the same as finding a hypergraph isomorphism,
    if it exists, which can be reduced in poly-time
    (but at the cost of increased space) to finding
    a graph isomorphism for bipartite graphs, as described in p1 of
    "Colored Hypergraph Isomorphism is Fixed Parameter Tractable".
    We use NetworkX and the VF2 algorithm to find the graph isomorphisms.
    Despite formidable worst-case complexity, this routine is fast in practice.

    Args:
        M: The first F_2 matrix.
        N: The second F_2 matrix.

    Returns:
        The hypergraph isomorphism, which is a bijective map on columns and
        corresponding bijective map on rows.
    """
    num_vertices_M = M.shape[0]
    num_edges_M = M.shape[1]

    num_vertices_N = N.shape[0]
    num_edges_N = N.shape[1]

    if num_vertices_M != num_vertices_N:
        return None
    if num_edges_M != num_edges_N:
        return None

    num_vertices = num_vertices_M
    num_edges = num_edges_M
    GM = nx.Graph()
    # red node = was originally a vertex in the hypergraph
    # green node = was originally a hyperedge in the hypergraph
    for i in range(num_vertices):
        GM.add_nodes_from([(i, {"color": "red"})])
    for j in range(num_edges):
        GM.add_nodes_from([(j + num_vertices, {"color": "green"})])
    for i in range(num_vertices):
        for j in range(num_edges):
            if M[i][j]:
                GM.add_edge(i, j + num_vertices)

    GN = nx.Graph()
    for i in range(num_vertices):
        GN.add_nodes_from([(i, {"color": "red"})])
    for j in range(num_edges):
        GN.add_nodes_from([(j + num_vertices, {"color": "green"})])
    for i in range(num_vertices):
        for j in range(num_edges):
            if N[i][j]:
                GN.add_edge(i, j + num_vertices)

    iso = nx.vf2pp_isomorphism(GM, GN, node_label="color")
    if iso is None:
        return None
    qubit_dict = {}
    syndrome_dict = {}
    for i in iso:
        if i < num_vertices:
            syndrome_dict[i] = iso[i]
        else:
            qubit_dict[i - num_vertices] = iso[i] - num_vertices

    return (qubit_dict, syndrome_dict)
