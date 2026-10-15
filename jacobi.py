import os
import sys

CWD=os.path.dirname(__file__)
# --- Local Deps --------------------------
sys.path.insert(1, CWD)
import grid
# --- Local Deps --------------------------

def init_solution(N: int) -> list[float]:
    if N < 2: raise ValueError("N must be >= 2")
    return [0] * (N+1)**2

if __name__ == "__main__":
    print("Hello world")
