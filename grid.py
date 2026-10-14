from typing import Tuple
from itertools import product
import math

# Docstring for file below functions

def uval(u: list[float], N: int, row: int, col: int) -> float:
    """Given (r,c), get value of u"""
    return u[uidx(N, row, col)]

def uidx(N: int, row: int, col: int) -> int:
    """Given (row,col) get index of flat u"""
    return row * (N + 1) + col

def calcN(ulen: int) -> int:
    """Calc N given h=1/N"""
    N = math.isqrt(ulen) - 1
    if (ulen < 4) or ((N+1)**2 != ulen):
        raise ValueError(f"impossible array length: {ulen}")
    return N

def h(N: int) -> float:
    """ h=1/N """
    if N <= 0: raise ValueError(f"Invalid N: {N}")
    return 1.0/float(N)

def coord(h: float, row: int, col: int) -> Tuple[float, float]:
    """(r,c) - > (x,y)"""
    return (float(col*h), float(row*h))

def interior_points(N: int) -> list[Tuple[int,int]]:
    """Returns list of (r,c) tuples"""
    return list(product(range(1,N), range(1,N)))

def bc_points(N: int) -> list[Tuple[float, float]]:
    """Returns a list of i,j tuples.
       Assumption, all bcs have the same boundary condition"""
    return [
        (row, col)
        for col, row in product(range(N + 1), repeat=2)
        if col == 0 or col == N or row == 0 or row == N
    ]


"""
Structured 2D Cartesian mesh for the unit square.

Domain: (x, y) in [0, 1] x [0, 1]
Discretization:
    N = number of intervals in each coordinate direction
    h = 1/N
    x = col*h, col = [0,N]
    y = row*h, row = [0,N]

Mesh points:
    Total:    (N+1)^2
    Interior: (N-1)^2
    Boundary: 4*N

Index conventions:
    row = y-direction index
    col = x-direction index

Boundary Conditions & Interior Points
    Boundary points have row or col equal to 0 or N.
    Interior points have 1 <= row,col <= N-1.
    BCs all set to u(∂Ω) = 0

Storage:
    The solution vector u is a flattened 1D array
    containing all mesh points, including boundaries.

    Row-major indexing: k = row*(N+1) + col

    Therefore: u[k] = u(col, row)

    Moving one column right increases k by 1.
    Moving one row up increases k by N+1.
"""


import unittest
import matplotlib.pyplot as plt

class TestMesh(unittest.TestCase):

    def test_spacing(self):
        self.assertEqual(h(4), 0.25)
        self.assertEqual(h(32), 1.0 / 32)

        with self.assertRaises(ValueError):
            h(0)

    def test_coordinates(self):
        self.assertEqual(coord(0.25, 0, 0), (0.0, 0.0))
        self.assertEqual(coord(0.25, 4, 4), (1.0, 1.0))
        self.assertEqual(coord(0.25, 2, 3), (0.75, 0.5))

    def test_calcN(self):
        self.assertEqual(calcN(4), 1)
        self.assertEqual(calcN(25), 4)
        self.assertEqual(calcN(1089), 32)

    def test_calcN_invalid(self):
        for length in (0, 3, 10, 1088):
            with self.assertRaises(ValueError):
                calcN(length)

    def test_interior_count(self):
        for N in (1, 2, 4, 8, 32):
            points = interior_points(N)
            self.assertEqual(len(points), (N - 1)**2)

    def test_boundary_count(self):
        for N in (1, 2, 4, 8, 32):
            points = bc_points(N)
            self.assertEqual(len(points), 4 * N)

    def test_no_duplicate_points(self):
        for N in (2, 4, 32):
            interior = interior_points(N)
            boundary = bc_points(N)

            self.assertEqual(len(interior), len(set(interior)))
            self.assertEqual(len(boundary), len(set(boundary)))

    def test_disjoint_sets(self):
        N = 32
        interior = set(interior_points(N))
        boundary = set(bc_points(N))

        self.assertEqual(interior & boundary, set())

    def test_complete_mesh(self):
        N = 32
        all_points = set(product(range(N + 1), repeat=2))
        generated = set(interior_points(N)) | set(bc_points(N))

        self.assertEqual(generated, all_points)

    def test_flattened_index(self):
        N = 4

        self.assertEqual(uidx(N, 0, 0), 0)
        self.assertEqual(uidx(N, 0, 4), 4)
        self.assertEqual(uidx(N, 1, 0), 5)
        self.assertEqual(uidx(N, 1, 1), 6)
        self.assertEqual(uidx(N, 2, 3), 13)
        self.assertEqual(uidx(N, 4, 4), 24)

    def test_index_uniqueness(self):
        N = 32
        indices = [
            uidx(N, row, col)
            for row, col in product(range(N + 1), repeat=2)
        ]

        self.assertEqual(len(indices), (N + 1)**2)
        self.assertEqual(set(indices), set(range((N + 1)**2)))

    # Adding bounds checking too slow
    #def test_invalid_index(self):
    #    for row, col in ((-1, 0), (0, -1), (5, 0), (0, 5)):
    #        with self.assertRaises(IndexError):
    #            uidx(4, row, col)

    def test_uval(self):
        N = 4
        u = [float(k) for k in range((N + 1)**2)]

        self.assertEqual(uval(u, N, 0, 0), 0.0)
        self.assertEqual(uval(u, N, 1, 1), 6.0)
        self.assertEqual(uval(u, N, 2, 3), 13.0)
        self.assertEqual(uval(u, N, 4, 4), 24.0)

def plot_mesh(N: int = 32) -> None:
    """Plot interior and boundary points of a 2D Cartesian mesh."""
    print(f"\n\n --> Plotting mesh for N = {N}")
    dx = h(N)

    # Generate physical coordinates
    interior = [
        coord(dx, row, col)
        for row, col in interior_points(N)
    ]

    boundary = [
        coord(dx, row, col)
        for row, col in bc_points(N)
    ]

    # Plot mesh
    fig, ax = plt.subplots(figsize=(7, 7))
    # Plot interior points
    if interior:
        xi, yi = zip(*interior)
        ax.scatter(xi, yi, c="black", s=8, label="Interior")

    # Plot boundary points
    xb, yb = zip(*boundary)
    ax.scatter(xb, yb, c="red", s=16, label="Boundary")

    # Meta
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"2D Cartesian Mesh (N={N})")

    ax.set_aspect("equal")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    unittest.main(exit=False)
    plot_mesh(N=32)
