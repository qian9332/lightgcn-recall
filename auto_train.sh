#!/bin/bash
# 自动分段训练 - 每次1个epoch,全量batch
cd /home/user/lightgcn-recall

for i in $(seq 1 80); do
    echo "=== Starting epoch run $i ==="
    python3 run_epoch.py 103 2>&1
    RC=$?
    if [ $RC -ne 0 ]; then
        echo "Error at run $i (exit code $RC)"
        break
    fi
    echo "=== Completed run $i ==="
done

echo "All training completed!"
cat logs/training.log
