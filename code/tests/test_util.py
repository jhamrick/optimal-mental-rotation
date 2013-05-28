import numpy as np
from util import safe_multiply, safe_log


def test_safe_log():
    arr = np.random.rand(10, 10) + 1
    log_arr1 = np.log(arr)
    log_arr2 = safe_log(arr)
    assert (log_arr1 == log_arr2).all()


def test_safe_multiply():
    arr1 = np.random.randn(10, 10)
    arr2 = np.random.randn(10, 10)
    prod1 = arr1 * arr2
    prod2 = safe_multiply(arr1, arr2)
    diff = np.abs(prod1 - prod2)
    if not (diff < 1e-8).all():
        print prod1
        print prod2
        raise ValueError
