.PHONY: figures pdf notes proposal project nips nips-workshop clean clean-figs clean-pdf

test:
	py.test --cov lib --cov-report html tests/

cython:
	python setup.py build_ext --inplace
