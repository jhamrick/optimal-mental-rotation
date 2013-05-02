import numpy as np


def similarity(I0, I1, sf=1):
    """Computes the similarity between images I0 and I1"""
    # C = log(1. / sqrt(2 * pi * sf**2))
    diff = np.exp(np.sum(-0.5 * (I0 - I1)**2 / (sf**2))) + 1
    return diff
