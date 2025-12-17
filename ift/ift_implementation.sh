#!/bin/bash
#SBATCH --job-name question+answer_charlotte
#SBATCH --chdir /users/lsimonnet/
#SBATCH --output /users/lsimonnet/meditron/axolotl_config/Legitron/reports/evaluation/Eval-%x.%j.out
#SBATCH --error /users/lsimonnet/meditron/axolotl_config/Legitron/reports/evaluation/Eval-%x.%j.err
#SBATCH --nodes 1             
#SBATCH --ntasks-per-node 1     
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 32      
#SBATCH --time 11:59:59     
#SBATCH --environment /users/lsimonnet/.edf/axolotl-vllm.toml
#SBATCH -A a127


# --- Environment Variables ---
export HF_HOME=/capstor/store/cscs/swissai/a127/homes/lsimonnet/hf_home
export HF_TOKEN="HF_TOKEN"
export CUDA_LAUNCH_BLOCKING=1
export NCCL_IB_DISABLE=1
echo "START TIME: $(date)"
set -eo pipefail
set -x

# --- Run the Evaluation ---
# Ensure your evaluate.py is in the current directory or provide full path
SCRIPT_PATH="/users/lsimonnet/meditron/axolotl_config/Legitron/ift_charlotte/ift_vllm.py"

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
