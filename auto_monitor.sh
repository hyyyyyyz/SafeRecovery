#!/bin/bash
# Auto-monitor and report experiment completion

LOG_FILE="/home/hurricane/RL/CSGLoco/experiment_status.log"

while true; do
    echo "=== $(date) ===" >> $LOG_FILE

    # Check ANYmal training progress
    ANYMAL_LOG="/home/hurricane/RL/CSGLoco/exp_anymal_multiseed.log"
    if [ -f "$ANYMAL_LOG" ]; then
        # Count completed trainings
        COMPLETED=$(grep -c "Done:" $ANYMAL_LOG 2>/dev/null || echo "0")
        echo "ANYmal multi-seed: $COMPLETED/15 trainings completed" >> $LOG_FILE
    fi

    # Check Vanilla/CaT @2000 progress
    VANILLA_LOG="/home/hurricane/RL/CSGLoco/exp_vanilla_cat_2000.log"
    if [ -f "$VANILLA_LOG" ]; then
        COMPLETED=$(grep -c "Done:" $VANILLA_LOG 2>/dev/null || echo "0")
        echo "Vanilla/CaT @2000: $COMPLETED/12 trainings completed" >> $LOG_FILE
    fi

    # Check rough terrain progress
    ROUGH_LOG="/home/hurricane/RL/CSGLoco/exp4_rough.log"
    if [ -f "$ROUGH_LOG" ]; then
        if grep -q "ALL DONE" $ROUGH_LOG; then
            echo "Rough terrain: COMPLETED" >> $LOG_FILE
        else
            LINES=$(wc -l < $ROUGH_LOG)
            echo "Rough terrain: Running ($LINES log lines)" >> $LOG_FILE
        fi
    fi

    # Check for new checkpoints
    NEW_CHECKPOINTS=$(find /home/hurricane/RL/CSGLoco/legged_gym/logs -name "model_1500.pt" -o -name "model_2000.pt" 2>/dev/null | wc -l)
    echo "Total checkpoints (1500/2000): $NEW_CHECKPOINTS" >> $LOG_FILE

    echo "---" >> $LOG_FILE

    # Check if all experiments are done
    ANYMAL_DONE=$(tmux capture-pane -t exp_anymal_multiseed -p 2>/dev/null | grep -c "All ANYmal multi-seed training complete" || echo "0")
    VANILLA_DONE=$(tmux capture-pane -t exp_vanilla_cat_2000 -p 2>/dev/null | grep -c "All 2000-iteration training complete" || echo "0")

    if [ "$ANYMAL_DONE" -gt 0 ] && [ "$VANILLA_DONE" -gt 0 ]; then
        echo "ALL TRAINING COMPLETED at $(date)" >> $LOG_FILE
        # Create completion flag
        touch /home/hurricane/RL/CSGLoco/ALL_TRAINING_COMPLETE
        break
    fi

    sleep 300  # Check every 5 minutes
done
