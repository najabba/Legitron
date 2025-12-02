#!/bin/bash
#SBATCH --job-name Eval_Llama
#SBATCH --chdir /users/$USER/
#SBATCH --output /users/$USER/Legitron/reports/evaluation/Eval-%x.%j.out
#SBATCH --error /users/$USER/Legitron/reports/evaluation/Eval-%x.%j.err
#SBATCH --nodes 1              
#SBATCH --ntasks-per-node 1     
#SBATCH --gres gpu:1            
#SBATCH --cpus-per-task 32      
#SBATCH --time 00:59:59     
#SBATCH --environment /users/$USER/.edf/axolotl.toml
#SBATCH -A a127

# --- Environment Variables ---
export HF_TOKEN= #Your Token

export CUDA_LAUNCH_BLOCKING=1
echo "START TIME: $(date)"
set -eo pipefail
set -x

# --- Run the Evaluation ---
# Ensure your evaluate.py is in the current directory or provide full path
SCRIPT_PATH="/users/$USER/Legitron/evaluation/evaluate.py"

echo "Running evaluation script..."

export CMD="python $SCRIPT_PATH $@"
echo $CMD

SRUN_ARGS=" \
--cpus-per-task $SLURM_CPUS_PER_TASK \
--jobid $SLURM_JOB_ID \
--wait 60 \
-A a127 \
--reservation=sai-a127
"

# We use 'python' directly instead of 'torchrun' because the script 
# handles its own model loading (device_map="auto")
srun $SRUN_ARGS bash -c "$CMD"

echo "END TIME: $(date)"
