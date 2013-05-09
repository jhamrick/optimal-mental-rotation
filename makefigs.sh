#!/bin/sh -e

# generate OmniGraffle figures
for fig in figures/*.graffle; do
    echo "Exporting '$fig'..."
    omnigraffle-export -f png "$fig" figures/
done
/usr/bin/osascript -e 'tell application "OmniGraffle Professional 5" to quit'

# now run analyses and generate figures from them
cd "code"
python ../run_nb.py notes.ipynb
