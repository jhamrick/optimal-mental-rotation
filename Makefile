.PHONY: figures pdf notes proposal clean-figs clean-pdf clean

all: figures pdf


figures:
	./makefigs.sh


pdf: notes proposal project
notes:
	./nb2pdf.sh notes
proposal:
	./tex2pdf.sh proposal
project:
	./tex2pdf.sh final-project

clean: clean-figs clean-pdf
clean-figs:
	rm -f figures/*.png
	rm -f figures/*.pdf
clean-pdf:
	rm -f man/*.pdf
	rm -rf man/notes_files/
	rm -rf man/proposal_files/
	rm -rf man/final-project_files/
