import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

CWD=os.path.dirname(__file__)
# --- Local Deps --------------------------
sys.path.insert(1, CWD)
import grid
# --- Local Deps --------------------------

def plot_solution(u: list[float], name: str, show_grid: bool = False) -> None:
    """Plot a scalar solution u(x,y) on the unit square.
        u: Flattened solution vector, including boundary nodes.
           Uses row-major ordering: k = row*(N+1) + col.
        show_grid: Whether to display the computational grid.
    """
    # Reconstruct 2D solution
    N = grid.calcN(len(u))
    U = np.asarray(u, dtype=float).reshape((N + 1, N + 1))

    # Generate coordinates
    x = np.linspace(0.0, 1.0, N + 1)
    y = np.linspace(0.0, 1.0, N + 1)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Interpolated contours
    contour = ax.contourf(X, Y, U, levels=50, cmap="viridis")

    fig.colorbar(contour, ax=ax, label="u(x, y)")

    # Optional computational grid
    if show_grid:
        for xi in x:
            ax.axvline(xi, color="black", linewidth=0.4, alpha=0.3)
        for yi in y:
            ax.axhline(yi, color="black", linewidth=0.4, alpha=0.3)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(name)
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()


def animate_solution(us: list[list[float]], name: str, show_grid: bool = False,
                     interval: int = 100, filename: str | None = None) -> FuncAnimation:
    """Animate a sequence of scalar solutions u(x,y) on the unit square.
    us: List of flattened solution vectors, including boundary nodes.
        Each vector uses row-major ordering:
        k = row*(N+1) + col.
    show_grid: Whether to display the computational grid.
    interval: Time between frames in milliseconds.
    filename: Optional output filename (.mp4 or .gif).
    """

    if not us:
        raise ValueError("us must contain at least one solution")

    N = grid.calcN(len(us[0]))
    expected = (N + 1)**2

    for i, u in enumerate(us):
        if len(u) != expected:
            raise ValueError(
                f"Frame {i}: Expected {expected} values, got {len(u)}"
            )

    # Reconstruct all 2D solutions
    U = np.asarray(us, dtype=float).reshape(
        (len(us), N + 1, N + 1)
    )

    # Generate coordinates
    x = np.linspace(0.0, 1.0, N + 1)
    y = np.linspace(0.0, 1.0, N + 1)
    X, Y = np.meshgrid(x, y)

    # Fixed color limits across all frames
    vmin = min(0.0, float(U.min()))
    vmax = max(0.0, float(U.max()))

    # Symmetric color scale around zero
    vmax_abs = max(abs(vmin), abs(vmax))
    if vmax_abs == 0:
        vmax_abs = 1.0

    levels = np.linspace(-vmax_abs, vmax_abs, 51)

    fig, ax = plt.subplots(figsize=(7, 6))

    contour = ax.contourf(
        X, Y, U[0], levels=levels, cmap="seismic"
    )

    fig.colorbar(contour, ax=ax, label="u(x, y)")

    # Optional computational grid
    if show_grid:
        for xi in x:
            ax.axvline(xi, color="black", linewidth=0.4, alpha=0.3)
        for yi in y:
            ax.axhline(yi, color="black", linewidth=0.4, alpha=0.3)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    title = ax.set_title(f"{name}")

    def update(frame: int):
        nonlocal contour

        # Remove previous contour collections
        contour.remove()

        # Draw updated solution
        contour = ax.contourf(
            X, Y, U[frame], levels=levels, cmap="seismic"
        )

        title.set_text(f"{name} | Iteration {frame}")

        return (contour, title)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(us),
        interval=interval,
        blit=False,
        repeat=False,
    )

    if filename is not None:
        fps = max(1, round(1000 / interval))

        if filename.endswith(".mp4"):
            anim.save(filename, writer=FFMpegWriter(fps=fps))
        elif filename.endswith(".gif"):
            anim.save(filename, writer=PillowWriter(fps=fps))
        else:
            raise ValueError("filename must end in .mp4 or .gif")

    plt.tight_layout()
    return anim


def plot_iterations(qoi: list[float], ylabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(range(len(qoi)), qoi, linewidth=1.5)

    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.1)

    plt.tight_layout()


if __name__ == "__main__":
    N = 64
    #plot_solution(N, j.init_solution(N), show_grid=True)

    plt.show()
