import math
from functional import seq

def init_solution(N: int) -> list[float]:
    if N < 2: raise ValueError("N must be >= 2")
    return [0] * (N+1)**2

# Vector Ops - Put in utils
def minus(a: float,b: float) -> float:
    return a-b

def plus(a: float, b: float) -> float:
    return a+b

def vop(v1: list[float], v2: list[float], op) -> list[float]:
    return list(
        map(op, zip(v1,v2))
    )

# Put in utils or driver directory
def residual(Au: list[float], f: list[float]) -> list[float]:
    """ r^k = f - Au^k    :        Au, f ∈ R^n"""
    return vop(f, Au, minus)

def L2prod(u: list[float], v: list[float]) -> float:
    if not (len(u) == len(v)): raise ValueError(f"len(u): {len(u)} neq len(v): {len(v)}")
    return seq(range(len(u))).map(lambda i : u[i] * v[i]).sum()

def L2norm(u: list[float]) -> float:
    l2_squared = seq(u).map(lambda ui: ui**2).sum()
    return math.sqrt(l2_squared)
