import os
import sys
import math
import numpy as np
from functional import seq
import matplotlib.pyplot as plt

CWD=os.path.dirname(__file__)
# --- Local Deps --------------------------
sys.path.insert(1, CWD)
import laplace as l
import grid as g
import plot as p
import cg
# --- Local Deps --------------------------

def val_cg():
    N = 32
    istop = 40
    u0 = l.init_u(N)
    f = l.init_f(N)

    u_true = l.generate_analytical_solution(N)
    soln = cg.cg_solve(u0,f,istop,nplot=1)
    solnN = soln[-1]
    print(l.verify_bcs(solnN))

    p.plot_solution(solnN, name=f"Conjugate Gradient: N: {N} - istop: {istop}", show_grid=True)
    p.plot_solution(u_true, name=f"Analytical Solution: N: {N}", show_grid=False)

def plot_log_roc(Ns, log_rocs, name):
    plt.subplots(figsize=(8, 8))
    plt.title(f"Rate of Convergence Dependence on h")
    plt.xlabel("$ln(N)$")
    plt.ylabel("$-ln(R_{40})$")

    logN = seq(Ns).map(lambda n: math.log(n)).to_list()
    def add_plot(lroc, name):
        plt.scatter(logN, lroc,label=name)
        slope, intercept = np.polyfit(logN, lroc, 1)
        plt.plot(logN, seq(logN).map(lambda n: n*slope + intercept).to_list(), "--")

    add_plot(log_rocs, name)
    plt.legend()

def log_roc(roc: float) -> float:
    return -1.0 * math.log(roc)

def hw_25():
    print("Running Cases for HW 2.5")
    Ns = [8,16,32]
    istop = 40

    u0s = [ l.init_u(N) for N in Ns ]
    fs = [ l.init_f(N) for N in Ns ]
    a_solns = [ l.generate_analytical_solution(N) for N in Ns ]

    solns = [
        cg.cg_solve(u0s[idx],fs[idx],istop,nplot=1)
        for idx in range(len(Ns))
    ]
    rocs = [
        l.calc_roc(solns[idx][-1], u0s[idx], a_solns[idx], istop)
        for idx in range(len(Ns))
    ]
    logrocs = [ log_roc(roc) for roc in rocs ]

    plot_log_roc(Ns, logrocs, name="Conjugate Gradient")

if __name__ == "__main__":
    print("Hello world")

    hw_25()

    plt.show()

