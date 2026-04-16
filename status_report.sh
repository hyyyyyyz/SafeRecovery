#!/bin/bash
# Final experiment status report

echo "=========================================="
echo "SafeRecovery Experiment Status Report"
echo "Time: $(date)"
echo "=========================================="
echo ""

# Check tmux sessions
echo "--- Active Tmux Sessions ---"
tmux ls 2>/dev/null || echo "No active sessions"
echo ""

# Check training logs
echo "--- Training Progress ---"
for log in exp_anymal_multiseed.log exp_vanilla_cat_2000.log; do
    if [ -f "/home/hurricane/RL/CSGLoco/$log" ]; then
        COMPLETED=$(grep -c "Done:" /home/hurricane/RL/CSGLoco/$log 2>/dev/null || echo "0")
        TOTAL=$(echo $log | grep -q "anymal" && echo "15" || echo "12")
        echo "$log: $COMPLETED/$TOTAL completed"
    fi
done
echo ""

# Check evaluation results
echo "--- Evaluation Results ---"
OUTDIR="/home/hurricane/RL/CSGLoco/safe_recovery_eval_v2"
if [ -d "$OUTDIR" ]; then
    echo "Total result files: $(ls $OUTDIR/*.json 2>/dev/null | wc -l)"

    echo ""
    echo "A1 6-seed results:"
    ls $OUTDIR/vanilla_seed*_standard.json 2>/dev/null | wc -l | xargs echo "  Vanilla:"
    ls $OUTDIR/cat_seed*_standard.json 2>/dev/null | wc -l | xargs echo "  CaT:"
    ls $OUTDIR/recovery_seed*_standard.json 2>/dev/null | wc -l | xargs echo "  Recovery:"

    echo ""
    echo "ANYmal results:"
    ls $OUTDIR/anymal_vanilla_seed*_standard.json 2>/dev/null | wc -l | xargs echo "  Vanilla:"
    ls $OUTDIR/anymal_cat_seed*_standard.json 2>/dev/null | wc -l | xargs echo "  CaT:"
    ls $OUTDIR/anymal_recovery_seed*_standard.json 2>/dev/null | wc -l | xargs echo "  Recovery:"

    echo ""
    echo "Training budget fairness (@2000):"
    ls $OUTDIR/vanilla_2000_seed*_standard.json 2>/dev/null | wc -l | xargs echo "  Vanilla @2000:"
    ls $OUTDIR/cat_2000_seed*_standard.json 2>/dev/null | wc -l | xargs echo "  CaT @2000:"

    echo ""
    echo "Rough terrain:"
    ls $OUTDIR/*_rough.json 2>/dev/null | wc -l | xargs echo "  Results:"
fi
echo ""

# Check completion flags
echo "--- Completion Status ---"
if [ -f "/home/hurricane/RL/CSGLoco/ALL_TRAINING_COMPLETE" ]; then
    echo "Training: COMPLETE"
else
    echo "Training: IN PROGRESS"
fi

if [ -f "/home/hurricane/RL/CSGLoco/ALL_EVAL_COMPLETE" ]; then
    echo "Evaluation: COMPLETE"
else
    echo "Evaluation: IN PROGRESS / PENDING"
fi
echo ""

# GPU status
echo "--- GPU Status ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
echo ""

echo "=========================================="
echo "End of Report"
echo "=========================================="
