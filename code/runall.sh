#!/bin/sh -e

python model_gs.py
python model_naive.py
python model_vm.py

for stim in `ls ../stimuli`; do
    python model_bq.py ${stim%.npz}
done
