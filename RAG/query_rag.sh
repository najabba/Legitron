#!/bin/bash
#SBATCH --job-name RAG_Query
#SBATCH --chdir /users/vesy/
#SBATCH --output /users/vesy/Legitron/reports/rag/Query-%x.%j.out
#SBATCH --error /users/vesy/Legitron/reports/rag/Query-%x.%j.err
#SBATCH --nodes 1              
#SBATCH --ntasks-per-node 1     
#SBATCH --gres gpu:1            
#SBATCH --cpus-per-task 32      
#SBATCH --time 01:00:00     
#SBATCH --environment /users/vesy/.edf/axolotl.toml
#SBATCH -A a127

# --- Environment Variables ---
export HF_TOKEN="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"

export CUDA_LAUNCH_BLOCKING=1
echo "START TIME: $(date)"
set -eo pipefail
set -x

# --- Run the RAG Query Script ---
SCRIPT_PATH="/users/vesy/Legitron/RAG/query_rag.py"

echo "Running RAG query script..."

# Default parameters (override with command line arguments)
INDEX_DIR="${INDEX_DIR:-/users/vesy/Legitron/RAG/ihl_index}"
LLM_MODEL="${LLM_MODEL:-/capstor/store/cscs/swissai/a127/homes/nabbassi/models/llama3.1-8B-IFT_charlotte}"
#"change this question to whatever you want"
QUESTION="${QUESTION:- help me I want to know in which rule specifically i can use to answer: Can we bother the elderly ?}"
TOP_K="${TOP_K:-5}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
TEMPERATURE="${TEMPERATURE:-0.7}"
OUTPUT_FILE="${OUTPUT_FILE:-/users/vesy/Legitron/reports/rag/output.txt}"

export CMD="python $SCRIPT_PATH \
  --index-dir $INDEX_DIR \
  --llm-model $LLM_MODEL \
  --question '$QUESTION' \
  --top-k $TOP_K \
  $@"

echo $CMD

SRUN_ARGS=" \
--cpus-per-task $SLURM_CPUS_PER_TASK \
--jobid $SLURM_JOB_ID \
--wait 60 \
-A a127 \
--reservation=sai-a127
"

srun $SRUN_ARGS bash -c "$CMD"

echo "END TIME: $(date)"
