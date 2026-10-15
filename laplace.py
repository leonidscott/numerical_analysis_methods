import os
import sys
from scipy.sparse import lil_array, csr_array

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

def f(x: float, y: float) -> float:
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

if __name__ == "__main__":
    print_A(4)
