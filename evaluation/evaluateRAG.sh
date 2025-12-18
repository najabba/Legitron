#!/bin/bash
#SBATCH --job-name=RAG_EVAL
#SBATCH --chdir /users/$USER/
#SBATCH --output /users/$USER/Legitron/reports/rag_eval/-%x.%j.out
#SBATCH --error /users/$USER/Legitron/reports/rag_eval/-%x.%j.err
#SBATCH --partition=debug
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 32
#SBATCH --time 01:30:00
#SBATCH --environment /users/$USER/.edf/axolotl.toml
#SBATCH -A a127

echo "====================="
echo "   RAG EVALUATION"
echo "   START: $(date)"
echo "====================="

set -eo pipefail
set -x

# -----------------------------------
# CONFIGURATION (EDIT IF NECESSARY)
# -----------------------------------
MODEL_PATH="/YOUR/MODEL/PATH"
INDEX_DIR="/users/$USER/Legitron/RAG/ihl_index"
SCRIPT_PATH="/users/$USER/Legitron/evaluation/evaluateRAG.py"


echo "Model path: $MODEL_PATH"
echo "Index dir:  $INDEX_DIR"
echo "Script:     $SCRIPT_PATH"

# -----------------------------------
# RUN EVALUATION
# -----------------------------------

CMD="python3 $SCRIPT_PATH \
    --model $MODEL_PATH \
    --index-dir $INDEX_DIR"

echo "Executing:"
echo "$CMD"

srun \
  --cpus-per-task $SLURM_CPUS_PER_TASK \
  --gres=gpu:1 \
  --jobid $SLURM_JOB_ID \
  --wait 60 \
  -A a127 \
  --reservation=sai-a127 \
  bash -c "$CMD"

echo "====================="
echo "   END: $(date)"
echo "====================="
