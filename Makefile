.PHONY: all test cython cogsci

all:

test:
	py.test --cov lib --cov-report html tests/

cython:
	python setup.py build_ext --inplace

cogsci:
	make cogsci -C man

clean:
	make clean -C man
