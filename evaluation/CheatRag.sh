#!/bin/bash
#SBATCH --job-name=Cheat_RAG
#SBATCH --chdir /users/$USER/
#SBATCH --output /users/$USER/Legitron/reports/cheatRag/ACheat%j.out
#SBATCH --error /users/$USER/Legitron/reports/cheatRag/ACheat%j.err
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 32
#SBATCH --time 01:30:00
#SBATCH --environment /users/$USER/.edf/axolotl.toml
#SBATCH -A a127

echo "====================="
echo "   Actual Cheat Rag"
echo "   START: $(date)"
echo "====================="

set -eo pipefail
set -x

# Have to do it once if chromdb is not installed. 
# If already ran one time, then comment both lines
srun --environment=/users/$USER/.edf/axolotl.toml python3 -m pip install --user autoawq

# -----------------------------------
# CONFIGURATION (EDIT IF NECESSARY)
# -----------------------------------

MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
INDEX_DIR="/users/$USER/Legitron/RAG/ihl_index"
SCRIPT_PATH="/users/$USER/Legitron/evaluation/CheatRag.py"


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