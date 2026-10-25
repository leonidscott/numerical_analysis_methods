import os
import sys
from scipy.sparse import lil_array, csr_array
from functional import seq
from itertools import product
import math

CWD=os.path.dirname(__file__)
# --- Local Deps --------------------------
sys.path.insert(1, CWD)
import grid
# --- Local Deps --------------------------

def gen_A(N: int) -> csr_array:
    """Generates an A matrix for the poisson equation ON THE INTERIOR POINTS ONLY.
       N_int: Interior number of points (per axis) on the domain: N_int = (N - 2)"""

    if N < 2: raise ValueError(f"N must have interior points (N >= 2), N={N}")

    N_int = N - 1
    n = N_int**2
    A = lil_array((n, n), dtype=float)
    h2 = grid.h(N)**2

    for row, col in grid.interior_points(N):
        k = (row - 1) * N_int + (col - 1)

        A[k, k] = 4.0 / h2                               # A_{k,k}    (Diag)
        if (col > 1):   A[k, k - 1] = -1.0 / h2          # A_{k,k-1}  (Left)
        if (col < N-1): A[k, k + 1] = -1.0 / h2          # A_{k,k+1}  (Right)
        if (row > 1):   A[k, k - N_int] = -1.0 / h2      # A_{k-1, k} (Down)
        if (row < N-1): A[k, k + N_int] = -1.0 / h2      # A_{k+1, k} (Up)

    return A.tocsr()

def rhs(x: float, y: float) -> float:
    """ RHS of Poisson Equation:

        f(x,y) = -[(6x-4)y(y-1) + 2x(x-1)^2]

    """
    return -1 * (
        ( (6*x - 4) * y  * (y - 1) )
        +
        ( 2*x  *  (x-1)**2)
    )

def analytical_solution(x: float, y: float) -> float:
    return (
        x * (x-1)**2 * y *(y-1)
    )

def print_A(N):
    A = gen_A(N)
    h2 = grid.h(N)**2
    print((A * h2).toarray())


def init_u(N: int) -> list[float]:
    if N < 2: raise ValueError("N must be >= 2")
    return [0] * (N+1)**2

def init_f(N: int) -> list[float]:
    if N < 2: raise ValueError("N must be >= 2")
    h = grid.h(N)
    all_coords= (
        seq(range(0,(N+1)**2))
        .map(lambda k: grid.k2rc(N,k))
        .map(lambda rc: grid.coord(h, rc[0], rc[1]))
        .to_list())
    f = seq(all_coords).map(lambda p: rhs(p[0],p[1])).to_list()
    # Respect BCs
    for r, c in grid.bc_points(N):
        f[grid.uidx(N, r, c)] = 0.0
    return f

def generate_analytical_solution(N: int) -> list[float]:
    u = init_u(N)
    h = grid.h(N)

    for r,c in product(range(N + 1), repeat=2):
        idx = grid.uidx(N,r,c)
        x, y = grid.coord(h,r,c)
        u[idx] = analytical_solution(x, y)

    return u

def verify_bcs(U: list[float]) -> bool:
    N = grid.calcN(len(U))
    bc_points = grid.bc_points(N)
    return (
        seq(bc_points)
        .map(lambda rc: grid.uidx(N, rc[0],rc[1]))
        .map(lambda k: U[k] == 0)
        .all())

def error_norm(Uk, Utrue) -> float:
    if len(Uk) != len(Utrue): raise ValueError(f"Uk and Utrue not of same length")
    kmax = len(Utrue)
    h = grid.h(grid.calcN(kmax))
    norm_squared = (
        seq(range(0,kmax))
        .map(lambda i: (Utrue[i] - Uk[i])**2 * h**2)
        .sum())
    return math.sqrt(norm_squared)

def calc_roc(Uk, U0, Utrue,i) -> float:
    "Rate of Convergece: -ln( ||Utrue - Uk||_2  / ||Utrue - U0|| ) / i (i = iteration)"
    rate_of_conv = -1.0 * math.log(error_norm(Uk, Utrue) / error_norm(Utrue, U0)) / i
    return rate_of_conv


if __name__ == "__main__":
    print_A(4)
