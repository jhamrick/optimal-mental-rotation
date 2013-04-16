#!/bin/sh -e

NAME=$1
CURPATH=`pwd`

if [ -z $NAME ]; then
    echo "Notebook name not provided"
    exit 1
fi

cd ../nbconvert
./nbconvert2.py latex_base "$CURPATH/$NAME.ipynb" > "$CURPATH/$NAME.tex"
cd "$CURPATH"
pdflatex "$NAME.tex"
