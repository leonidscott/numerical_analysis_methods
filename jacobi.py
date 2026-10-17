import os
import sys
from functional import seq, pseq

CWD=os.path.dirname(__file__)
# --- Local Deps --------------------------
sys.path.insert(1, CWD)
import grid as g
# --- Local Deps --------------------------

def jsolve(u0: list[float], f: list[float], istop: int) -> list[float]:
    N = g.calcN(len(u0))
    h = g.h(N)

    U = u0.copy()

    uidx = (lambda  r,c : g.uidx(N,r,c))
    inter_points = g.interior_points(N)
    def delta(point) -> float:
        """ δ_{r,c} for 2nd order central scheme"""
        r,c = point
        stencil = U[uidx(r,c+1)] + U[uidx(r,c-1)] + U[uidx(r+1,c)] + U[uidx(r-1,c)] - 4*U[uidx(r,c)]
        res = f[uidx(r,c)] + (stencil / h**2)
        return (h**2 / 4.0) * res

    for i in range(1,istop+1):
        print(f"  iteration: {i}")
        #U^{n+1} = U^{n} + \delta
        # Step 1: Calc Delta
        deltas = seq(inter_points).map(delta).to_list()

        # Step 2: Apply Update
        for d in range(0,len(deltas)):
            r,c = inter_points[d]
            k = uidx(r,c)
            U[k] += deltas[d]

    return U

if __name__ == "__main__":
    print("Hello world")
