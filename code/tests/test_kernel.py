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


def check_K2(x, kernel, mll, params):
    kK = kernel(x, x)
    mK = np.array([[mll.K(params, x1-x2) for x2 in x] for x1 in x])
    diff = np.abs(kK - mK)
    if not (diff < thresh).all():
        print diff
        raise AssertionError("incorrect kernel function outputs")


######################################################################


def test_gaussian_K():
    x = np.linspace(-2*np.pi, 2*np.pi, 16)
    for i in xrange(N_big):
        h = rand_h()
        w = rand_w()
        kernel = gaussian_kernel(h, w, jit=False)
        mll = KernelMLL('gaussian', h=h, w=w, s=0)
        yield (check_K, x, kernel, mll)


def test_gaussian_K2():
    x = np.linspace(-2*np.pi, 2*np.pi, 16)
    for i in xrange(N_big):
        h = rand_h()
        w = rand_w()
        kernel = gaussian_kernel(h, w, jit=False)
        mll = KernelMLL('gaussian', h=None, w=None, s=0)
        yield (check_K2, x, kernel, mll, (h, w))


def test_periodic_K():
    x = np.linspace(-2*np.pi, 2*np.pi, 16)
    for i in xrange(N_big):
        h = rand_h()
        w = rand_w()
        p = rand_p()
        kernel = periodic_kernel(h, w, p, jit=False)
        mll = KernelMLL('periodic', h=h, w=w, p=p, s=0)
        yield (check_K, x, kernel, mll)


def test_periodic_K2():
    x = np.linspace(-2*np.pi, 2*np.pi, 16)
    for i in xrange(N_big):
        h = rand_h()
        w = rand_w()
        p = rand_p()
        kernel = periodic_kernel(h, w, p, jit=False)
        mll = KernelMLL('periodic', h=None, w=None, p=None, s=0)
        yield (check_K2, x, kernel, mll, (h, w, p))


def test_kernel_params_h():
    h = None
    w = rand_w()
    p = rand_p()
    s = 0
    mll = KernelMLL('periodic', h=h, w=w, p=p, s=s)
    assert mll.kernel_params((1,)) == (1, w, p, s)


def test_kernel_params_w():
    h = rand_h()
    w = None
    p = rand_p()
    s = 0
    mll = KernelMLL('periodic', h=h, w=w, p=p, s=s)
    assert mll.kernel_params((1,)) == (h, 1, p, s)


def test_kernel_params_p():
    h = rand_h()
    w = rand_w()
    p = None
    s = 0
    mll = KernelMLL('periodic', h=h, w=w, p=p, s=s)
    assert mll.kernel_params((1,)) == (h, w, 1, s)


def test_kernel_params_s():
    h = rand_h()
    w = rand_w()
    p = rand_p()
    s = None
    mll = KernelMLL('periodic', h=h, w=w, p=p, s=s)
    assert mll.kernel_params((1,)) == (h, w, p, 1)
