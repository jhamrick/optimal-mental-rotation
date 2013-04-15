.PHONY: figures pdf

all: figures pdf

figures:
	omnigraffle-export -f png figures/causal-structure.graffle figures/
	omnigraffle-export -f png figures/mental-image.graffle figures/
	/usr/bin/osascript -e 'tell application "OmniGraffle Professional 5" to quit'

pdf:
	nbconvert --format latex mental-rotation.ipynb
	pdflatex mental-rotation.tex
	pdflatex mental-rotation.tex

clean:
	rm -f figures/causal-structure.png
	rm -f figures/mental-image.png
	rm -rf mental_rotation_files/
	rm -f mental-rotation.aux
	rm -f mental-rotation.log
	rm -f mental-rotation.out
	rm -f mental-rotation.pdf
	rm -f mental-rotation.tex
