import numpy as np
np.seterr(all='raise')

from kernel import KernelMLL
from snippets.stats import gaussian_kernel, periodic_kernel

######################################################################

N_big = 20
N_small = 5
thresh = 1e-6

######################################################################


def rand_h():
    h = np.random.uniform(0, 2)
    return h


def rand_w():
    w = np.random.uniform(np.pi / 32., np.pi / 2.)
    return w


def rand_p():
    p = np.random.uniform(0.33, 3)
    return p


def check_K(x, kernel, mll):
    kK = kernel(x, x)
    mK = np.array([[mll.K(x1-x2) for x2 in x] for x1 in x])
    diff = np.abs(kK - mK)
    if not (diff < thresh).all():
        print diff
        raise AssertionError("incorrect kernel function outputs")


def test_gaussian_K():
    x = np.linspace(-2*np.pi, 2*np.pi, 16)
    for i in xrange(N_big):
        h = rand_h()
        w = rand_w()
        kernel = gaussian_kernel(h, w, jit=False)
        mll = KernelMLL('gaussian', h=h, w=w, s=0)
        yield (check_K, x, kernel, mll)


def test_periodic_K():
    x = np.linspace(-2*np.pi, 2*np.pi, 16)
    for i in xrange(N_big):
        h = rand_h()
        w = rand_w()
        p = rand_p()
        kernel = periodic_kernel(h, w, p, jit=False)
        mll = KernelMLL('periodic', h=h, w=w, p=p, s=0)
        yield (check_K, x, kernel, mll)
