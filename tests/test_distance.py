import numpy as np

from ssip.basic_functions import find_X_basis, flip_code
from ssip.code_examples import (
    GTri_code,
    QRM_code,
    Shor_B,
    Shor_code,
    Steane_code,
    Surface2_code,
    Surface3_code,
    Toric_code,
    Twist_code,
)
from ssip.distance import (
    code_distance,
    distance_GAP,
    distance_lower_bound_z3,
    distance_z3,
)


def test_code_distance():
    # should be invariant under permutation of PZ, PX
    assert distance_z3(Shor_code) == 3
    assert distance_z3(GTri_code) == 2
    assert distance_z3(Toric_code) == 3
    assert distance_z3(QRM_code) == 3
    assert distance_z3(Twist_code) == 3
    assert distance_z3(Steane_code) == 3
    assert distance_z3(Surface3_code) == 3
    assert distance_z3(Surface2_code) == 2

    assert code_distance(Shor_code, True) == 3
    assert code_distance(GTri_code) == 2
    assert code_distance(Toric_code) == 3


def test_gap_distance():
    num_sets = 100
    assert distance_GAP(Shor_code, None, num_sets) == 3
    assert distance_GAP(flip_code(Shor_code), None, num_sets) == 3

    assert distance_GAP(GTri_code, None, num_sets) >= 2

    assert distance_GAP(Toric_code, None, num_sets) == 3

    assert distance_GAP(flip_code(QRM_code), None, num_sets) >= 3

    assert distance_GAP(Twist_code, None, num_sets) == 3
    assert distance_GAP(Steane_code, None, num_sets) == 3
    assert distance_GAP(Surface3_code, None, num_sets) == 3
    assert distance_GAP(Surface2_code, None, num_sets) == 2


def test_lower_bound():
    Lx = np.array(find_X_basis(Shor_code))
    assert distance_lower_bound_z3(Shor_B, Lx, 2) is None
    assert distance_lower_bound_z3(Shor_B, Lx, 3) is not None
    assert distance_lower_bound_z3(Shor_B, Lx, 4) is not None
