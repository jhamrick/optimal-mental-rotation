.PHONY: figures pdf

all: figures pdf

figures:
	./makefigs.sh

notes:
	./nb2pdf.sh notes

proposal:
	./tex2pdf.sh proposal

clean:
	rm -f figures/*.png
	rm -f man/*.pdf
	rm -rf man/notes_files/
	rm -rf man/proposal_files/
