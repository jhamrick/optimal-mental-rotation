.PHONY: clean

cogsci-2014.pdf: cogsci-2014.tex
	pdflatex cogsci-2014.tex
	bibtex cogsci-2014
	pdflatex cogsci-2014.tex
	pdflatex cogsci-2014.tex

clean: 
	rm -f *.aux
	rm -f *.bbl
	rm -f *.blg
	rm -f *.dvi
	rm -f *.log
	rm -f *.pdf
