import os
import sys
from functional import seq, pseq

CWD=os.path.dirname(__file__)
# --- Local Deps --------------------------
sys.path.insert(1, CWD)
import grid as g
from utils import L2norm
# --- Local Deps --------------------------

def jsolve(u0: list[float], f: list[float], istop: int, nplot: int = 0) -> tuple[list[float], list]:
    N = g.calcN(len(u0))
    h = g.h(N)

    U = u0.copy()
    nplots = []

    uidx = (lambda  r,c : g.uidx(N,r,c))
    inter_points = g.interior_points(N)
    def residual(point) -> float:
        """ residual_{r,c} for 2nd order central scheme res=(f - Au)"""
        r,c = point
        stencil = U[uidx(r,c+1)] + U[uidx(r,c-1)] + U[uidx(r+1,c)] + U[uidx(r-1,c)] - 4*U[uidx(r,c)]
        res = f[uidx(r,c)] + (stencil / h**2)
        return res

    for i in range(1,istop+1):
        print(f"  iteration: {i}", end="")
        #U^{n+1} = U^{n} + \delta
        # Step 1: Calc Delta - δ_{r,c} = (h**2/4.0)*res_{r,c}
        res = seq(inter_points).map(residual).to_list()
        deltas = seq(res).map(lambda r: (h**2 / 4.0) * r).to_list()

        # Step 2: Apply Update
        for d in range(0,len(deltas)):
            r,c = inter_points[d]
            k = uidx(r,c)
            U[k] += deltas[d]

        # Stats 'n stuff
        norm_res = L2norm(res) / L2norm(f)
        print(f",    residual: {norm_res}")
        if (nplot!=0 and i%nplot == 0):
            if(nplot != 1): print("Plotting solution")
            nplots.append(U.copy())

    return U, nplots

if __name__ == "__main__":
    print("Hello world")
