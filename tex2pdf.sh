#!/bin/bash -e

base="$1"
suffix="_files"
filedir="$base$suffix"

if [ -z "$base" ]; then
    echo "Base name not provided"
    exit 1
fi

echo "Converting 'man/$base.tex'"

cd man
pdflatex "$base" && bibtex "$base" && pdflatex "$base" && pdflatex "$base"

if [ ! -d "$filedir" ]; then
    echo "$filedir"
    mkdir "$filedir"
fi

mv "$base."* "$filedir/"
mv "texput.log" "$filedir/" || true
cp "$filedir/$base.pdf" .
cp "$filedir/$base.tex" .
