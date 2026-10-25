import os
import sys

CWD=os.path.dirname(__file__)
# --- Local Deps --------------------------
sys.path.insert(1, CWD)
import grid as g
import utils as v
from utils import L2prod, L2norm
# --- Local Deps --------------------------

def central_stencil(u: list[float], point:tuple[float,float]) -> float:
    r,c = point
    N = g.calcN(len(u))
    h = g.h(N)
    uidx = (lambda  r,c : g.uidx(N,r,c))

    stencil = (u[uidx(r,c+1)] + u[uidx(r,c-1)] + u[uidx(r+1,c)] + u[uidx(r-1,c)]
               - 4*u[uidx(r,c)])
    return (-1.0 * stencil / h**2)

def Au(u: list[float]) -> list[float]:
    N = g.calcN(len(u))
    Au = [0.0] * len(u)

    for r, c in g.interior_points(N):
        k = g.uidx(N, r, c)
        Au[k] = central_stencil(u, (r, c))

    return Au

def cg_solve(u0: list[float], f: list[float], istop: int, nplot: int = 0) -> list:
    print(f"Running Conjugate Gradient with istop: {istop}")
    snapshots = []
    uk = u0.copy()
    N = g.calcN(len(u0))

    # Checks & Prep
    if len(f) != len(u0): raise ValueError("u0 and f dimensions differ")
    for r, c in g.bc_points(N):
        f[g.uidx(N, r, c)] = 0.0


    Auk = Au(uk)
    gk = [Auk[i] - f[i] for i in range(len(f))]
    dk = [-gi for gi in gk]
    gamma_k = L2prod(gk,gk)

    for i in range(1,istop+1):
        print(f"  iteration: {i}",end="")
        qk = Au(dk)
        alpha_k = gamma_k / L2prod(dk,qk)

        uk = [
            uk[k] + alpha_k * dk[k]
            for k in range(len(uk))
        ]

        gk = [
            gk[k] + alpha_k * qk[k]
            for k in range(len(gk))
        ]


        gamma_old = gamma_k
        gamma_k = L2prod(gk,gk)
        beta_k = gamma_k/gamma_old

        dk = [
            -gk[k] + beta_k * dk[k]
            for k in range(len(dk))
        ]

        # Residual Calculation
        Auk = Au(uk)
        res_vec = [ f[idx] - Auk[idx] for idx in range(len(Auk)) ]
        res_val = L2norm(res_vec)
        print(f", residual: {res_val}")

        if ( (nplot!=0 and i%nplot == 0) or (i == istop) ):
            snapshots.append(uk.copy())
    return snapshots
