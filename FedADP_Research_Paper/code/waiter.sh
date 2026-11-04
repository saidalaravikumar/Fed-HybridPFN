while pgrep -f "queue3.sh" > /dev/null; do sleep 30; done
(python3 experiments_v3.py abl3 0 2 >> ../results/v3a0.log 2>&1; echo DONE >> ../results/v3a0.log) &
(python3 experiments_v3.py abl3 1 2 >> ../results/v3a1.log 2>&1; echo DONE >> ../results/v3a1.log) &
wait
