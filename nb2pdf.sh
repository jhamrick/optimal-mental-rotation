#!/bin/sh -e

base=$1
cd man
curpath=`pwd`
suffix="_files"
filedir="$base$suffix"

if [ -z "$base" ]; then
    echo "Notebook name not provided"
    exit 1
fi

echo "Converting '$curpath/$base.ipynb'"

if [ ! -d "$filedir" ]; then
    mkdir "$filedir"
fi
cd $HOME/project/nbconvert
./nbconvert2.py latex_base "$curpath/$base.ipynb" > "$curpath/$filedir/$base.tex"

cd "$curpath"
rm "$base.tex"
cp "references.bib" "$filedir"

cd "$filedir"
pdflatex "$base" && bibtex "$base" && pdflatex "$base" && pdflatex "$base"
cp "$base.pdf" ../
