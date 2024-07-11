from dataclasses import dataclass

import numpy as np

from ssip.basic_functions import (
    BinMatrix,
    CSScode,
    column_echelon_form,
    find_paired_basis,
    find_X_basis,
    find_Z_basis,
    image_basis_calc,
    kernel_basis_calc,
    multiply_F2,
    vec_addition,
)


@dataclass
class MergeResult:
    """A collection of data which informs the user about a merge, or
    multiple merges, which were performed.

    Attributes:
        Code: The output merged code.
        MergeMap: The map on qubits from the input to output code.
        NewZStabs: The indices of any new Z stabiliser generators.
        NewXStabs: The indices of any new X stabiliser generators.
        NewQubits: The indices of any new data qubits.
        NewZLogicals: A basis for the space of new Z logicals
        introduced by the merge, if there are any.
        NewXLogicals: A basis for the space of new X logicals
        introduced by the merge, if there are any.
        OldZLogicals: A basis for the space of old Z logicals
        retained by the merge, if there are any.
        OldXLogicals: A basis for the space of old X logicals
        retained by the merge, if there are any.

    """

    Code: CSScode
    # if the merge was a Z-merge, this is the chain map at degree 1;
    # if X-merge, cochain map at degree 1
    MergeMap: BinMatrix
    NewZStabs: list
    NewXStabs: list
    NewQubits: list
    NewZLogicals: list
    NewXLogicals: list
    OldZLogicals: list
    OldXLogicals: list


def find_logical_splitting(
    old_code: CSScode,
    new_code: CSScode,
    merge_map: BinMatrix,
    num_before_logicals: int,
    num_after_logicals: int,
    num_merges: int = 1,
    basis: str = "Z",
):
    if basis != "Z" and basis != "X":
        raise ValueError("Must enter Z or X basis.")
    num_new_logicals = num_after_logicals - num_before_logicals + num_merges
    hom_before = None
    if basis == "Z":
        hom_before = np.array(find_Z_basis(old_code)).T
    else:
        hom_before = np.array(find_X_basis(old_code)).T
    if hom_before.shape[1] != num_before_logicals:
        raise RuntimeError("Dimensions do not match up when returning merge data.")
    hom_before_in_merged_code = multiply_F2(merge_map, hom_before)

    im_basis_after = None
    if basis == "Z":
        im_basis_after = np.array(image_basis_calc(new_code.PZ.T))
    else:
        im_basis_after = np.array(image_basis_calc(new_code.PX.T))
    Q = np.array(kernel_basis_calc(im_basis_after))

    ker_before_in_after_hom = multiply_F2(Q, hom_before_in_merged_code)

    mat1 = column_echelon_form(ker_before_in_after_hom)

    num_old_logicals = num_before_logicals - num_merges
    vecs_to_add = [0] * num_old_logicals
    height = ker_before_in_after_hom.shape[0]
    for i in np.arange(num_old_logicals):
        v = [0] * (mat1.shape[0] - height)
        for j in np.arange(height, mat1.shape[0]):
            v[j - height] = mat1[j][i]
        vecs_to_add[i] = v

    logical_basis = [0] * num_old_logicals
    for i in np.arange(num_old_logicals):
        vecs = [
            list(hom_before_in_merged_code[:, j])
            for j in np.arange(len(vecs_to_add[i]))
            if vecs_to_add[i][j]
        ]

        logical_basis[i] = vec_addition(vecs)

    S = np.array(kernel_basis_calc(ker_before_in_after_hom.T))

    if basis == "Z":
        ker_after = np.array(kernel_basis_calc(new_code.PX)).T
    else:
        ker_after = np.array(kernel_basis_calc(new_code.PZ)).T
    ker_after_quotient_old_logicals = multiply_F2(S, multiply_F2(Q, ker_after))
    mat = column_echelon_form(ker_after_quotient_old_logicals, True)

    height = ker_after_quotient_old_logicals.shape[0]
    vecs_to_add = [0] * num_new_logicals
    for i in np.arange(num_new_logicals):
        v = [0] * (mat.shape[0] - height)
        for j in np.arange(height, mat.shape[0]):
            v[j - height] = mat[j][i]
        vecs_to_add[i] = v

    gauge_basis = [0] * num_new_logicals
    for i in np.arange(num_new_logicals):
        vecs = [
            list(ker_after[:, j])
            for j in np.arange(len(vecs_to_add[i]))
            if vecs_to_add[i][j]
        ]
        gauge_basis[i] = vec_addition(vecs)

    if basis == "Z":
        new_Z_logicals = gauge_basis
        old_Z_logicals = logical_basis
        total_Z_basis = gauge_basis + logical_basis

        trial_X_basis = find_X_basis(new_code)
        total_X_basis = find_paired_basis(total_Z_basis, trial_X_basis)
        new_X_logicals = total_X_basis[:num_new_logicals]
        old_X_logicals = total_X_basis[num_new_logicals:]

        return (new_Z_logicals, new_X_logicals, old_Z_logicals, old_X_logicals)
    else:
        new_X_logicals = gauge_basis
        old_X_logicals = logical_basis
        total_X_basis = gauge_basis + logical_basis

        trial_Z_basis = find_Z_basis(new_code)
        total_Z_basis = find_paired_basis(total_X_basis, trial_Z_basis)
        new_Z_logicals = total_Z_basis[:num_new_logicals]
        old_Z_logicals = total_Z_basis[num_new_logicals:]

        return (new_Z_logicals, new_X_logicals, old_Z_logicals, old_X_logicals)
