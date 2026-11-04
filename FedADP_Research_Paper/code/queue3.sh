W=$1
for S in main3 abl3 fact3 tfm3 tau3 donors3 epsc3 kappa3; do python3 experiments_v3.py $S $W 2 >> ../results/v3q$W.log 2>&1; done
echo DONE >> ../results/v3q$W.log
