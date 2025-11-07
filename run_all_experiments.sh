#!/bin/bash
# Run all Deep Learning Experiments (6-10)

echo "======================================"
echo "Running All Deep Learning Experiments"
echo "======================================"
echo ""

export CUDA_VISIBLE_DEVICES=""

for exp in exp6 exp7 exp8 exp9 exp10; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Running $exp.py"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    uv run python ${exp}.py
    echo ""
    echo "Completed $exp.py"
    echo ""
done

echo "======================================"
echo "All Experiments Completed!"
echo "======================================"
