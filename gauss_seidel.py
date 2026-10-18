import os
import sys
from functional import seq, pseq

CWD=os.path.dirname(__file__)
# --- Local Deps --------------------------
sys.path.insert(1, CWD)
import grid as g
from utils import L2norm
# --- Local Deps --------------------------

def gs_row(u0: list[float], f: list[float], istop: int, nplot: int = 0) -> tuple[list[float],list]:
    N = g.calcN(len(u0))
    h = g.h(N)

    U = u0.copy()
    nplots = []

    uidx = (lambda  r,c : g.uidx(N,r,c))
    def residual(r,c) -> float:
        """ residual_{r,c} for 2nd order central scheme res=(f - Au)"""
        stencil = U[uidx(r,c+1)] + U[uidx(r,c-1)] + U[uidx(r+1,c)] + U[uidx(r-1,c)] - 4*U[uidx(r,c)]
        res = f[uidx(r,c)] + (stencil / h**2)
        return res

    for i in range(1,istop+1):
        print(f"  iteration: {i}", end="")
        iter_res = []
        for r in range(1,N):
            #U^{n+1} = U^{n} + \delta

            # Step 1: Calc Delta - δ_{r,c} = (h**2/4.0)*res_{r,c}
            res = seq(range(1,N)).map(lambda c: residual(r,c)).to_list()
            deltas = seq(res).map(lambda rs: (h**2 / 4.0) * rs).to_list()

            # Step 2: Apply Update
            for dix in range(0,len(deltas)):
                c = dix + 1 # column
                U[uidx(r,c)] += deltas[dix]

            # Stats n stuff
            iter_res += res
        # Stats n stuff
        norm_res = L2norm(iter_res) / L2norm(f)
        print(f",    residual: {norm_res}")
        if (nplot!=0 and i%nplot == 0):
            if(nplot != 1): print("Plotting solution")
            nplots.append(U.copy())

    return U,nplots

def gs_diag(u0: list[float], f: list[float], istop: int, nplot: int = 0) -> tuple[list[float],list]:
    N = g.calcN(len(u0))
    h = g.h(N)

    U = u0.copy()
    nplots = []

    uidx = (lambda  r,c : g.uidx(N,r,c))
    def residual(r,c) -> float:
        """ residual_{r,c} for 2nd order central scheme res=(f - Au)"""
        stencil = U[uidx(r,c+1)] + U[uidx(r,c-1)] + U[uidx(r+1,c)] + U[uidx(r-1,c)] - 4*U[uidx(r,c)]
        res = f[uidx(r,c)] + (stencil / h**2)
        return res

    for i in range(1,istop+1):
        print(f"  iteration: {i}", end="")
        iter_res = []
        for r in range(1,N):
            for c in range(1,N):
                #U^{n+1} = U^{n} + \delta

                # Step 1: Calc Delta - δ_{r,c} = (h**2/4.0)*res_{r,c}
                res = seq(range(1,N)).map(lambda c: residual(r,c)).to_list()
                res = residual(r,c)
                iter_res.append(res)
                delta = (h**2 / 4.0) * res

                # Step 2: Apply Update
                U[uidx(r,c)] += delta
        norm_res = L2norm(iter_res) / L2norm(f)
        print(f",    residual: {norm_res}")
        if (nplot!=0 and i%nplot == 0):
            if(nplot != 1): print("Plotting solution")
            nplots.append(U.copy())

    return U,nplots


if __name__ == "__main__":
    print("Hello world")
