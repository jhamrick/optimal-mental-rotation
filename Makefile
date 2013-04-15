all:
	nbconvert --format latex mental-rotation.ipynb
	pdflatex mental-rotation.tex
	pdflatex mental-rotation.tex

clean:
	rm -rf mental_rotation_files/
	rm -f mental-rotation.aux
	rm -f mental-rotation.log
	rm -f mental-rotation.out
	rm -f mental-rotation.pdf
	rm -f mental-rotation.tex
