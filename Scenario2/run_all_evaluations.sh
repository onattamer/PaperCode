#!/bin/bash

# Paths to evaluation scripts
EVAL_SCRIPT="evaluate.py"
BID_SCRIPT="bidmetric.py"

# Common config
NUM_EXPERIMENTS=10

# List of matchups
MATCHUPS=("t1vst1" "t1vst2" "t1vst3" "t2vst2" "t2vst3" "t3vst3")

# --- Run evaluate3.py for each matchup ---
for matchup in "${MATCHUPS[@]}"; do
    echo "🔍 Evaluating $matchup with evaluate3.py"

    python "$EVAL_SCRIPT" \
        --result_dir "$matchup" \
        --output_dir "eval_$matchup" \
        --num_experiments $NUM_EXPERIMENTS
done

# --- Then run bidmetric.py for each matchup ---
for matchup in "${MATCHUPS[@]}"; do
    echo "🔍 Running bidmetric.py for $matchup"

    python "$BID_SCRIPT" \
        --result_dir "$matchup" \
        --output_dir "eval_$matchup" \
        --num_experiments $NUM_EXPERIMENTS
done

echo "✅ All evaluations and bidmetric computations completed!"
