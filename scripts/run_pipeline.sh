#!/bin/bash
set -e

source ~/.bashrc
conda activate dacslab_ckim310

echo "🚀 Starting LoRA Fine-tuning..."
python src/fine_tuning.py

echo "🧪 Evaluating model..."
python src/evaluate.py > logs/eval.log

ACC=$(grep "Validation accuracy" logs/eval.log | awk '{print $3}')

if (( $(echo "$ACC > $ACCURACY_THRESHOLD" | bc -l) )); then
  echo "✅ Accuracy ($ACC) passed threshold. Deploying model..."
  python src/push_to_localhub.py
else
  echo "⚠️ Accuracy ($ACC) below threshold. Skipping deploy."
fi
