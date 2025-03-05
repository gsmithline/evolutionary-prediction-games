import numpy as np

def discrete_replicator(p0, fitness, T, return_welfare=False):
    """
    Discrete replicator equation dynamics
    Taylor & Jonker, 1978, eq. (3)
    """
    p = np.array(p0)
    out = [p]
    welfare = []
    for t in range(T):
        f = fitness(p)+1
        p = (p*f)/(p@f)
        out.append(p)
        welfare.append(p@(f-1))
    if not return_welfare:
        return np.vstack(out)
    else:
        return np.vstack(out), np.array(welfare)