#!/bin/sh
W=$1
for S in main ablation gates disjoint eps eps_credit kappa nlab ens shift mrule msweep scale phase; do
  python3 experiments.py $S $W 2 >> ../results/q$W.log 2>&1
done
echo DONE >> ../results/q$W.log
