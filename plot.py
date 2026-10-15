import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

CWD=os.path.dirname(__file__)
# --- Local Deps --------------------------
sys.path.insert(1, CWD)
import grid
import jacobi as j
import laplace
# --- Local Deps --------------------------

def plot_solution(N: int, u: list[float], show_grid: bool = False) -> None:
    """Plot a scalar solution u(x,y) on the unit square.
        N: Number of grid intervals per coordinate direction.
        u: Flattened solution vector, including boundary nodes.
           Uses row-major ordering: k = row*(N+1) + col.
        show_grid: Whether to display the computational grid.
    """
    if len(u) != (N + 1)**2:
        raise ValueError(f"Expected {(N+1)**2} values, got {len(u)}")

    # Reconstruct 2D solution
    U = np.asarray(u, dtype=float).reshape((N + 1, N + 1))

    # Generate coordinates
    x = np.linspace(0.0, 1.0, N + 1)
    y = np.linspace(0.0, 1.0, N + 1)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Interpolated contours
    contour = ax.contourf(X, Y, U, levels=50, cmap="seismic")

    fig.colorbar(contour, ax=ax, label="u(x, y)")

    # Optional computational grid
    if show_grid:
        for xi in x:
            ax.axvline(xi, color="black", linewidth=0.4, alpha=0.3)
        for yi in y:
            ax.axhline(yi, color="black", linewidth=0.4, alpha=0.3)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Poisson Solution (N={N})")
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()

def generate_analytical_solution(N: int) -> list[float]:
    u = j.init_solution(N)
    h = grid.h(N)

    for r,c in product(range(N + 1), repeat=2):
        idx = grid.uidx(N,r,c)
        x, y = grid.coord(h,r,c)
        u[idx] = laplace.analytical_solution(x, y)

    return u


if __name__ == "__main__":
    N = 64
    #plot_solution(N, j.init_solution(N), show_grid=True)
    true_soln = generate_analytical_solution(N)
    plot_solution(N, true_soln, show_grid=False)

    plt.show()
