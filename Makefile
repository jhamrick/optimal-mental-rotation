.PHONY: figures pdf

all: figures pdf

figures:
	./makefigs.sh

pdf:
	./makepdf.sh mental-rotation

clean:
	rm -f figures/*.png
	rm -rf mental-rotation_files/
	rm -f mental-rotation.pdf
