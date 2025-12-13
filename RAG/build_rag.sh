#!/bin/bash
#SBATCH --job-name Build_RAG
#SBATCH --chdir /users/vesy/
#SBATCH --output /users/vesy/Legitron/reports/rag/Build-%x.%j.out
#SBATCH --error /users/vesy/Legitron/reports/rag/Build-%x.%j.err
#SBATCH --nodes 1              
#SBATCH --ntasks-per-node 1     
#SBATCH --gres gpu:1            
#SBATCH --cpus-per-task 32      
#SBATCH --time 00:30:00     
#SBATCH --environment /users/vesy/.edf/axolotl.toml
#SBATCH -A a127

# Have to do it once if chromdb is not installed
#srun --environment=/users/vesy/.edf/axolotl.toml python3 -m pip install --user chromadb
#srun --environment=/users/vesy/.edf/axolotl.toml python3 -m pip install --user sentence-transformers

# --- Environment Variables ---
export CUDA_LAUNCH_BLOCKING=1
echo "START TIME: $(date)"
set -eo pipefail
set -x

# --- Run the Build RAG Script ---
SCRIPT_PATH="/users/vesy/Legitron/RAG/build_rag.py"

echo "Building RAG FAISS index..."

# Default parameters (override with command line arguments)
RULES_FILE="${RULES_FILE:-/users/vesy/Legitron/RAG/rules_with_interpretations.json}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-BAAI/bge-large-en}"
OUTPUT_DIR="${OUTPUT_DIR:-/users/vesy/Legitron/RAG/ihl_index}"

export CMD="python $SCRIPT_PATH \
  --rules $RULES_FILE \
  --model $EMBEDDING_MODEL \
  --outdir $OUTPUT_DIR \
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
echo "RAG index built at: $OUTPUT_DIR"
