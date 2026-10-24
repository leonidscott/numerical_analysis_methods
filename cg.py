import os
import sys
from functional import seq, pseq

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
    return (stencil / h**2)

def gfn(u: list[float],f:list[float], point:tuple[float,float]) -> float:
    """ residual_{r,c} for 2nd order central scheme res=(f - Au)"""
    uidx = (lambda  r,c : g.uidx(N,r,c))
    _,c = point

    Au = central_stencil(u,point)
    return Au - f[uidx(f,c)]

def cg_solve(u0: list[float], f: list[float], istop: int, nplot: int = 0) -> list:
    print(f"Running Conjugate Gradient with istop: {istop}")
    Us = []
    uk = u0.copy()

    N = g.calcN(len(u0))
    inter_points = seq(g.interior_points(N))
    gk = (inter_points.map(lambda pt: gfn(uk,f,pt)).to_list())
    dk = (seq(gk).map(lambda gki: -1.0 * gki).to_list())
    gamma_k = L2norm(gk)

    for i in range(1,istop+1):
        print(f"  iteration: {i}")
        qk = (inter_points.map(lambda pt: central_stencil(dk,pt)).to_list())
        alpha_k = gamma_k / L2prod(dk,qk)
        uk = (seq(range(len(uk))).map(lambda i : uk[i] + alpha_k * dk[i]).to_list())
        gk = (seq(range(len(gk))).map(lambda i : gk[i] + alpha_k * qk[i]).to_list())

        gamma_old = gamma_k
        gamma_k = L2prod(gk,gk)
        beta = gamma_k/gamma_old

        dk = (seq(range(len(gk))).map(lambda i : -1.0 * gk[i] + beta * dk[i]))

        if (i == istop-1 or (nplot!=0 and i%nplot == 0)):
            Us.append(uk.copy())
    return Us
