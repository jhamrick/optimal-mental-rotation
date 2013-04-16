.PHONY: figures pdf

all: figures pdf

figures:
	./makefigs.sh

pdf:
	./makepdf.sh mental-rotation

clean:
	rm -f figures/causal-structure.png
	rm -f figures/mental-image.png
	rm -rf mental_rotation_files/
	rm -f mental-rotation.pdf
	rm -f mental-rotation.tex
	rm -f *.aux
	rm -f *.out
	rm -f *.log
