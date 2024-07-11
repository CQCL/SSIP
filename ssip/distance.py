import shutil
import subprocess
from ast import literal_eval
from functools import reduce
from pathlib import Path

import numpy as np

from ssip.basic_functions import (
    BinMatrix,
    CSScode,
    compose_to_zero,
    find_X_basis,
    find_Z_basis,
    num_data_qubits,
    systolic_distance,
)


def Z_distance(C: CSScode) -> int:
    """Finds the Z distance of a CSS code in a naive manner, by enumerating over
    all Z logicals.

    Args:
        C: The CSScode.


    Return:
        The Z distance of C.
    """
    return systolic_distance(C.PZ.T, C.PX)


def X_distance(C: CSScode) -> int:
    """Finds the X distance of a CSS code in a naive manner, by enumerating over
    all X logicals.

    Args:
        C: The CSScode.


    Return:
        The X distance of C.
    """
    return systolic_distance(C.PX.T, C.PZ)


def code_distance(C: CSScode, verbose: bool = False) -> int:
    """Finds the distance of a CSS code in a naive manner, by enumerating over
    all logicals.

    Args:
        C: The CSScode.


    Return:
        The distance of C.
    """
    dZ = Z_distance(C)
    if verbose:
        print(f"{dZ=}")
    dX = X_distance(C)
    if verbose:
        print(f"{dX=}")
    return min(dZ, dX)


# taken almost verbatim from Simon Burton's Qumba, see https://github.com/punkdit/qumba
def distance_lower_bound_z3(Hx: BinMatrix, Lx: BinMatrix, d: int) -> list | None:
    """Checks to see whether there are any Z logicals of weight at most d. If there are,
    returns the first one found.

    Args:
        Hx: The X parity check matrix of the code.
        Lx: A spanning set of X logicals of the code.
        d: The weight to check below.


    Return:
        The first nontrivial Z logical of weight at most d, if there is one.
        Otherwise returns None.
    """
    from z3 import Bool, If, Not, Or, Solver, Sum, Xor, sat

    m, n = Hx.shape
    k, n1 = Lx.shape

    solver = Solver()
    add = solver.add
    v = [Bool("v%d" % i) for i in range(n)]

    term = Sum([If(v[i], 1, 0) for i in range(n)]) == d
    add(term)

    def check(hx):
        terms = [v[j] for j in range(n) if hx[j]]
        if len(terms) > 1:
            return reduce(Xor, terms)
        elif len(terms) == 1:
            return terms[0]
        raise RuntimeError("dead check")

    # parity checks
    for i in range(m):
        add(Not(check(Hx[i])))

    # non-trivial logical
    term = reduce(Or, [check(Lx[i]) for i in range(k)])
    add(term)

    result = solver.check()
    if result != sat:
        return

    model = solver.model()
    v = [model.evaluate(v[i]) for i in range(n)]
    v = [int(literal_eval(str(vi))) for vi in v]

    assert compose_to_zero(Hx, np.array([v]).T), "bug bug... try updating z3?"
    assert not compose_to_zero(Lx, np.array([v]).T), "bug bug... try updating z3?"
    assert sum(v) == d, ".sum(v)==%d: bug bug... try updating z3?" % sum(v)
    return v


def distance_z3(
    C: CSScode, Lz: BinMatrix | None = None, Lx: BinMatrix | None = None
) -> int:
    """Finds the distance of a CSS code. Searches for logicals with weight 1, then
    weight 2 and so on. Works best for codes with low distances, but is implemented
    using Z3 so is quite fast.

    Args:
        C: The CSScode.
        Lz: A spanning set of Z logicals of the code.
        Lx: A spanning set of X logicals of the code.


    Return:
        The distance of C.
    """
    n = num_data_qubits(C)
    if Lz is None:
        Lz = np.array(find_Z_basis(C))
    if Lx is None:
        Lx = np.array(find_X_basis(C))

    d_X = 1
    while d_X < n:
        v = distance_lower_bound_z3(C.PZ, Lz, d_X)
        if v is not None:
            break
        d_X += 1

    d_Z = 1
    while d_Z < d_X:
        v = distance_lower_bound_z3(C.PX, Lx, d_Z)
        if v is not None:
            break
        d_Z += 1

    return min(d_Z, d_X)


# GAP QDistRnd I/O hack
def distance_GAP(
    C: CSScode,
    gap_path: str | None = None,
    num_information_sets: int | None = None,
    input_filename: str | None = None,
    output_filename: str | None = None,
) -> int:
    """Gives an upper bound on the distance of a CSS code using GAP. If the code is small
    enough and the number of information sets is high then this upper bound is tight.
    Empirically, we find that setting num_information_sets = 10,000 gives
    complete accuracy at ~200 qubits.

    Args:
        C: The CSScode.
        gap_path: The absolute path to the GAP directory.
        num_information_sets: The number of information sets to make.
        input_filename: Which file to use as input to GAP.
        output_filename: Which file to receive results from GAP.


    Return:
        An upper bound on the distance of C.
    """
    if input_filename is None:
        input_filename = "input.g"
    if output_filename is None:
        output_filename = "output.txt"
    if gap_path is None:
        gap_path = "gap"
    if num_information_sets is None:
        # empirically, 10,000 gives complete accuracy at ~200 qubits
        num_information_sets = 10000

    # Do the first way round...
    with Path.open(Path(input_filename), "w") as f:
        f.write('LoadPackage("QDistRnd");; \nF:=GF(2);;\n')
        str_PX = "Hx:=One(F)*["
        for i in range(C.PX.shape[0]):
            str_row = "["
            int_row = list(C.PX[i])
            for entry in int_row:
                str_row += str(int(entry)) + ","
            str_row = str_row[:-1] + "],"
            str_PX += str_row
        str_PX = str_PX[:-1] + "];;\n"
        f.write(str_PX)

        str_PZ = "Hz:=One(F)*["
        for i in range(C.PZ.shape[0]):
            str_row = "["
            int_row = list(C.PZ[i])
            for entry in int_row:
                str_row += str(int(entry)) + ","
            str_row = str_row[:-1] + "],"
            str_PZ += str_row
        str_PZ = str_PZ[:-1] + "];;\n"
        f.write(str_PZ)

        f.write(
            "d:= DistRandCSS(Hz,Hx,"
            + str(num_information_sets)
            + ',0,2 : field:=F);; \nPrintTo("'
            + output_filename
            + '", d);; \nquit;'
        )
        f.close()

    subprocess.run(
        [
            str(shutil.which(gap_path)),
            "--nointeract",
            "--norepl",
            input_filename,
        ],
        shell=False,  # noqa: S603
        stdout=subprocess.DEVNULL,
    )

    d1 = 0
    with Path.open(Path(output_filename)) as f:
        d1 = int(f.read())

    # Do the other way round...
    with Path.open(Path(input_filename), "w") as f:
        f.write('LoadPackage("QDistRnd");; \nF:=GF(2);;\n')
        str_PX = "Hx:=One(F)*["
        for i in range(C.PX.shape[0]):
            str_row = "["
            int_row = list(C.PX[i])
            for entry in int_row:
                str_row += str(int(entry)) + ","
            str_row = str_row[:-1] + "],"
            str_PX += str_row
        str_PX = str_PX[:-1] + "];;\n"
        f.write(str_PX)

        str_PZ = "Hz:=One(F)*["
        for i in range(C.PZ.shape[0]):
            str_row = "["
            int_row = list(C.PZ[i])
            for entry in int_row:
                str_row += str(int(entry)) + ","
            str_row = str_row[:-1] + "],"
            str_PZ += str_row
        str_PZ = str_PZ[:-1] + "];;\n"
        f.write(str_PZ)

        f.write(
            "d:= DistRandCSS(Hx,Hz,"
            + str(num_information_sets)
            + ',0,2 : field:=F);; \nPrintTo("'
            + output_filename
            + '", d);; \nquit;'
        )
        f.close()

    subprocess.run(
        [
            str(shutil.which(gap_path)),
            "--nointeract",
            "--norepl",
            input_filename,
        ],
        shell=False,  # noqa: S603
        stdout=subprocess.DEVNULL,
    )

    d2 = 0
    with Path.open(Path(output_filename)) as f:
        d2 = int(f.read())

    return min(d1, d2)


# Hack to use CSS min distance to find subsystem CSS dressed distance
def subsystem_distance_GAP(
    C: CSScode,
    gauge_Zs: list[list[int]],
    gauge_Xs: list[list[int]],
    gap_path: str | None = None,
    num_information_sets: int | None = None,
    input_filename: str | None = None,
    output_filename: str | None = None,
) -> int:
    """Gives an upper bound on the dressed distance of a subsystem CSS code using GAP.
    If the code is small enough and the number of information sets is high then this
    upper bound is tight.

    Args:
        C: The CSScode.
        gauge_Zs: A spanning set of the gauged Z logicals.
        gauge_Xs: A spanning set of the gauged X logicals.
        gap_path: The absolute path to the GAP directory.
        num_information_sets: The number of information sets to make.
        input_filename: Which file to use as input to GAP.
        output_filename: Which file to receive results from GAP.


    Return:
        An upper bound on the dressed distance of C as a subsystem code.
    """
    if input_filename is None:
        input_filename = "input.g"
    if output_filename is None:
        output_filename = "output.txt"
    if gap_path is None:
        gap_path = "gap"
    if num_information_sets is None:
        # empirically, 10,000 typically gives complete accuracy at ~200 qubits
        num_information_sets = 10000

    # Do the first way round...
    with Path.open(Path(input_filename), "w") as f:
        f.write('LoadPackage("QDistRnd");; \nF:=GF(2);;\n')
        str_PX = "Hx:=One(F)*["
        for i in range(C.PX.shape[0]):
            str_row = "["
            int_row = list(C.PX[i])
            for entry in int_row:
                str_row += str(int(entry)) + ","
            str_row = str_row[:-1] + "],"
            str_PX += str_row
        str_PX = str_PX[:-1] + "];;\n"
        f.write(str_PX)

        str_PZ = "Hz:=One(F)*["
        for i in range(C.PZ.shape[0]):
            str_row = "["
            int_row = list(C.PZ[i])
            for entry in int_row:
                str_row += str(int(entry)) + ","
            str_row = str_row[:-1] + "],"
            str_PZ += str_row
        for gauge_logical in gauge_Zs:
            str_row = "["
            for entry in gauge_logical:
                str_row += str(int(entry)) + ","
            str_row = str_row[:-1] + "],"
            str_PZ += str_row
        str_PZ = str_PZ[:-1] + "];;\n"
        f.write(str_PZ)

        f.write(
            "d:= DistRandCSS(Hz,Hx,"
            + str(num_information_sets)
            + ',0,2 : field:=F);; \nPrintTo("'
            + output_filename
            + '", d);; \nquit;'
        )
        f.close()

    subprocess.run(
        [
            str(shutil.which(gap_path)),
            "--nointeract",
            "--norepl",
            input_filename,
        ],
        shell=False,  # noqa: S603
        stdout=subprocess.DEVNULL,
    )

    d1 = 0
    with Path.open(Path(output_filename)) as f:
        d1 = int(f.read())

    # Do the other way round...
    with Path.open(Path(input_filename), "w") as f:
        f.write('LoadPackage("QDistRnd");; \nF:=GF(2);;\n')
        str_PX = "Hx:=One(F)*["
        for i in range(C.PX.shape[0]):
            str_row = "["
            int_row = list(C.PX[i])
            for entry in int_row:
                str_row += str(int(entry)) + ","
            str_row = str_row[:-1] + "],"
            str_PX += str_row
        for gauge_logical in gauge_Xs:
            str_row = "["
            for entry in gauge_logical:
                str_row += str(int(entry)) + ","
            str_row = str_row[:-1] + "],"
            str_PX += str_row
        str_PX = str_PX[:-1] + "];;\n"
        f.write(str_PX)

        str_PZ = "Hz:=One(F)*["
        for i in range(C.PZ.shape[0]):
            str_row = "["
            int_row = list(C.PZ[i])
            for entry in int_row:
                str_row += str(int(entry)) + ","
            str_row = str_row[:-1] + "],"
            str_PZ += str_row
        str_PZ = str_PZ[:-1] + "];;\n"
        f.write(str_PZ)

        f.write(
            "d:= DistRandCSS(Hx,Hz,"
            + str(num_information_sets)
            + ',0,2 : field:=F);; \nPrintTo("'
            + output_filename
            + '", d);; \nquit;'
        )
        f.close()

    subprocess.run(
        [
            str(shutil.which(gap_path)),
            "--nointeract",
            "--norepl",
            input_filename,
        ],
        shell=False,  # noqa: S603
        stdout=subprocess.DEVNULL,
    )

    d2 = 0
    with Path.open(Path(output_filename)) as f:
        d2 = int(f.read())

    return min(d1, d2)
