from itertools import product
import os
import sys
import math
from functional import seq
import matplotlib.pyplot as plt

CWD=os.path.dirname(__file__)
# --- Local Deps --------------------------
sys.path.insert(1, CWD)
import grid
import laplace
from jacobi import jsolve
from plot import animate_solution, plot_iterations, plot_solution
# --- Local Deps --------------------------

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
    f = seq(all_coords).map(lambda p: laplace.f(p[0],p[1])).to_list()
    return f

def generate_analytical_solution(N: int) -> list[float]:
    u = init_u(N)
    h = grid.h(N)

    for r,c in product(range(N + 1), repeat=2):
        idx = grid.uidx(N,r,c)
        x, y = grid.coord(h,r,c)
        u[idx] = laplace.analytical_solution(x, y)

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
        .map(lambda i: (Utrue[i] - Uk[i])**2 * h)
        .sum())
    return math.sqrt(norm_squared)

def calc_roc(Uk, U0, Utrue,i) -> float:
    "Rate of Convergece: -ln( ||Utrue - Uk||_2  / ||Utrue - U0|| ) / i (i = iteration)"
    rate_of_conv = -1.0 * math.log(error_norm(Uk, Utrue) / error_norm(Uk, U0)) / i
    return rate_of_conv

if __name__ == "__main__":
    print("Hello world")
    N = 32
    istop = 200
    u0 = init_u(N)
    f = init_f(N)

    # Solve
    U,nplots  = jsolve(u0, f, istop, nplot=1)

    # Plot Analytical Solution
    true_soln = generate_analytical_solution(N)
    plot_solution(true_soln, name=f"Analytical: {N}", show_grid=False)


    # Plot Approximated Solution
    plot_solution(U, name=f"Jacobi: {N} - istop: {istop}", show_grid=True)
    #anim = animate_solution(nplots, name=f"Jacobi: {N}", show_grid=True,
    #                        filename=CWD+f"/imgs/jacobi-N={N}-istop={istop}.mp4")

    # Verify Boundary Conditions are correct
    assert(verify_bcs(U))
    print(f"BCs consistent? {verify_bcs(U)}")

    # Plot Error & Rate of Convergence
    error_k = seq(nplots).map(lambda uk : error_norm(uk, true_soln)).to_list()
    plot_iterations(error_k, "error_norm", "error norm")

    roc = (
        seq(range(1,len(nplots))) # Prevent division by zero error when i = 0
        .map(lambda i: calc_roc(nplots[i], u0, true_soln, i))
        .to_list()
    )
    plot_iterations(roc, "$R_k$", f"Rate of Convergence for Jacobi - N={N}")

    plt.show()
