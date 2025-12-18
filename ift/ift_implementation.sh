#!/bin/bash
#SBATCH --job-name question+answer_charlotte
#SBATCH --chdir /users/$USER/
#SBATCH --output /users/$USER/Legitron/reports/data_gen/-%x.%j.out
#SBATCH --error /users/$USER/Legitron/reports/data_gen/-%x.%j.err
#SBATCH --nodes 1             
#SBATCH --ntasks-per-node 1     
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 32      
#SBATCH --time 11:59:59     
#SBATCH --environment /users/$USER/.edf/axolotl-vllm.toml
#SBATCH -A a127


# --- Environment Variables ---
export HF_HOME=/capstor/store/cscs/swissai/a127/homes/$USER/hf_home
export HF_TOKEN= #Your Token
export CUDA_LAUNCH_BLOCKING=1
export NCCL_IB_DISABLE=1
echo "START TIME: $(date)"
set -eo pipefail
set -x

# --- Run the Scripts ---
SCRIPT_PATH="/users/$USER/Legitron/ift_charlotte/ift_vllm.py"

echo "Running Data Generation script..."

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
