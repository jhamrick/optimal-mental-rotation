#!/bin/sh -e

for fig in figures/*.graffle; do
    echo "Exporting '$fig'..."
    omnigraffle-export -f png "$fig" figures/
done

/usr/bin/osascript -e 'tell application "OmniGraffle Professional 5" to quit'
