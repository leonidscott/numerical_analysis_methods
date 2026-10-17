from itertools import product
import os
import sys
from functional import seq
import matplotlib.pyplot as plt

CWD=os.path.dirname(__file__)
# --- Local Deps --------------------------
sys.path.insert(1, CWD)
import grid
import laplace
from jacobi import jsolve
from plot import plot_solution
# --- Local Deps --------------------------

def init_u(N: int) -> list[float]:
    if N < 2: raise ValueError("N must be >= 2")
    return [0] * (N+1)**2

def init_f(N: int) -> list[float]:
    if N < 2: raise ValueError("N must be >= 2")
    all_points = seq(range(0,(N+1)**2)).map(lambda k: grid.k2rc(N,k)).to_list()
    f = seq(all_points).map(lambda p: laplace.f(p[0],p[1])).to_list()
    return f

def generate_analytical_solution(N: int) -> list[float]:
    u = init_u(N)
    h = grid.h(N)

    for r,c in product(range(N + 1), repeat=2):
        idx = grid.uidx(N,r,c)
        x, y = grid.coord(h,r,c)
        u[idx] = laplace.analytical_solution(x, y)

    return u


if __name__ == "__main__":
    print("Hello world")
    N = 32
    istop = 32
    u0 = init_u(N)
    f = init_f(N)
    U  = jsolve(u0, f, istop)
    plot_solution(N, U, name=f"Jacobi: {N} - istop: {istop}", show_grid=True)

    true_soln = generate_analytical_solution(N)
    plot_solution(N, true_soln, name=f"Analytical: {N}", show_grid=False)
    plt.show()
