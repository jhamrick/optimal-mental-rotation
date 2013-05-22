import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
np.seterr(all='raise')

from kernel import KernelMLL, cholesky
from snippets.stats import GP, gaussian_kernel, periodic_kernel

######################################################################

N_big = 20
N_small = 1
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


def check_L(mat0):
    L = cholesky(mat0)
    mat = np.dot(L, L.T)
    diff = np.abs(mat0 - mat)
    if not (diff < thresh).all():
        print mat0
        print mat
        assert False


def check_K_inv(K):
    L = cholesky(K)
    Li = np.linalg.inv(L)
    Ki = np.dot(Li.T, Li)
    I = np.dot(K, Ki)
    diff = np.abs(I - np.eye(I.shape[0]))
    assert (diff < thresh).all()


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


def test_gaussian_make_kernel():
    x = np.linspace(-2*np.pi, 2*np.pi, 16)
    for i in xrange(N_big):
        h = rand_h()
        w = rand_w()
        mll = KernelMLL('gaussian', h=h, w=w, s=0)
        kernel = mll.make_kernel(params=(h, w, 1, 0), jit=False)
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


def test_periodic_K2():
    x = np.linspace(-2*np.pi, 2*np.pi, 16)
    for i in xrange(N_big):
        h = rand_h()
        w = rand_w()
        p = rand_p()
        kernel = periodic_kernel(h, w, p, jit=False)
        mll = KernelMLL('periodic', h=None, w=None, p=None, s=0)
        yield (check_K2, x, kernel, mll, (h, w, p))


def test_periodic_make_kernel():
    x = np.linspace(-2*np.pi, 2*np.pi, 16)
    for i in xrange(N_big):
        h = rand_h()
        w = rand_w()
        p = rand_p()
        mll = KernelMLL('periodic', h=h, w=w, p=p, s=0)
        kernel = mll.make_kernel(params=(h, w, p, 0), jit=False)
        yield (check_K, x, kernel, mll)


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


def check_params(p0, mll, x, y):
    p = mll.maximize(x, y, verbose=True, ntry=5)
    xx = np.linspace(-2*np.pi, 2*np.pi, 100)
    yy = np.sin(xx)
    mu, cov = GP(mll.make_kernel(params=p, jit=False), x, y, xx)
    plt.plot(x, y, 'ro')
    plt.plot(xx, mu, 'r-')
    plt.plot(xx, yy, 'k-')
    plt.show()
    diff = np.abs(np.array(p) - np.array(p0))
    if not (diff < thresh).all():
        print p
        print p0
        raise ValueError("bad parameters")


def test_maximize():
    x = np.linspace(-2*np.pi, 2*np.pi, 16)
    y = np.sin(x)
    for i in xrange(N_small):
        h0 = None
        w0 = None
        p0 = 1
        s0 = 0
        mll = KernelMLL('gaussian', h=h0, w=w0, p=p0, s=s0)
        yield check_params, (1, 1, p0, s0), mll, x, y


def test_cholesky():
    n = 10
    for i in xrange(N_big):
        x = np.random.rand(n) * 10
        d = x[:, None] - x[None, :]
        mat0 = scipy.stats.norm.pdf(d)
        yield check_L, mat0


def test_K_inv():
    x = np.linspace(-2*np.pi, 2*np.pi, 16)
    for i in xrange(N_big):
        h = rand_h()
        w = rand_w()
        mll = KernelMLL('gaussian', h=h, w=w, s=0)
        kernel = mll.make_kernel(params=(h, w, 1, 0), jit=False)
        K = kernel(x, x)
        yield (check_K_inv, K)
