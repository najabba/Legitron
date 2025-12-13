#!/bin/bash
#SBATCH --job-name=RAG_EVAL
#SBATCH --chdir /users/vesy/
#SBATCH --output /users/vesy/Legitron/reports/rag_eval/RAG_EVAL-%x.%j.out
#SBATCH --error /users/vesy/Legitron/reports/rag_eval/RAG_EVAL-%x.%j.err
#SBATCH --partition=debug
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 32
#SBATCH --time 01:30:00
#SBATCH --environment /users/vesy/.edf/axolotl.toml
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
#"meta-llama/Meta-Llama-3-8B-Instruct"
#"/capstor/store/cscs/swissai/a127/homes/nabbassi/models/llama3.1-8B-IFT_charlotte"
MODEL_PATH="/capstor/store/cscs/swissai/a127/homes/nabbassi/models/llama3.1-8B-IFT_charlotte"
INDEX_DIR="/users/vesy/Legitron/RAG/ihl_index"
SCRIPT_PATH="/users/vesy/Legitron/evaluation/evaluateRAG.py"


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