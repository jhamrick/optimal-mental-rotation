.PHONY: figures pdf notes proposal project nips nips-workshop clean clean-figs clean-pdf

all: figures pdf


figures:
	./makefigs.sh


pdf: nips-workshop
notes:
	./nb2pdf.sh notes
proposal:
	./tex2pdf.sh proposal
project:
	./tex2pdf.sh final-project
nips:
	./tex2pdf.sh nips-2013
nips-workshop:
	./tex2pdf.sh nips-2013-bayesian-optimization

clean: clean-figs clean-pdf
clean-figs:
	rm -f figures/*.png
	rm -f figures/*.pdf
clean-pdf:
	rm -f man/*.pdf
	rm -rf man/notes_files/
	rm -rf man/proposal_files/
	rm -rf man/final-project_files/
	rm -rf man/nips-2013_files/
	rm -rf man/nips-2013-bayesian-optimization_files/

test:
	py.test --cov lib --cov-report html tests/
