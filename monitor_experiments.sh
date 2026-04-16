#!/bin/bash
# Monitor experiment progress on remote server

echo "=== Experiment Monitor ==="
echo "Time: $(date)"
echo ""

echo "--- Tmux Sessions ---"
tmux ls 2>/dev/null || echo "No tmux sessions"
echo ""

echo "--- Log Files ---"
ls -la /home/hurricane/RL/CSGLoco/*.log 2>/dev/null | tail -10
echo ""

echo "--- Recent Log Activity ---"
for log in exp4_rough.log exp_anymal_multiseed.log exp_vanilla_cat_2000.log; do
    if [ -f "/home/hurricane/RL/CSGLoco/$log" ]; then
        echo "--- $log (last 5 lines) ---"
        tail -5 "/home/hurricane/RL/CSGLoco/$log"
        echo ""
    fi
done

echo "--- Checkpoints Created (last 10) ---"
find /home/hurricane/RL/CSGLoco/legged_gym/logs -name "model_*.pt" -mmin -60 2>/dev/null | head -10

echo ""
echo "--- GPU Usage ---"
nvidia-smi 2>/dev/null || echo "nvidia-smi not available"

echo ""
echo "=== End Monitor ==="
