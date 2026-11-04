W=$1
for S in gmsc20 calib6 prune6; do python3 experiments_v6.py $S $W 2 >> ../results/v6q$W.log 2>&1; done
echo DONE >> ../results/v6q$W.log
