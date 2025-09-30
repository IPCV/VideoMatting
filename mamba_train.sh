#!/bin/bash
#SBATCH -J MAMBA-Stg1                         # Job name
#SBATCH -N 1                                  # Number of nodes
#SBATCH -n 80                                 # Number of tasks (CPUs)


#SBATCH --chdir=/gpfs/scratch/ehpc176/        # Working directory
#SBATCH --gres=gpu:4                          # Request 4 GPUs
#SBATCH --time=3-00:00:00                     # Max execution time
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#SBATCH -o Experiments/Mamba/exp_00/stage1/stage1.out           # STDOUT
#SBATCH -e Experiments/Mamba/exp_00/stage1/stage1.err           # STDERR
#SBATCH --mail-type=all
#SBATCH --mail-user=sergi.garcia@upf.edu

# Load necessary modules
module load singularity  # Load Singularity if it's not already available
module load cuda/12.6

# Set environment variables for distributed training
export MASTER_ADDR=$(hostname)  # Set master node
export MASTER_PORT=36492        # Default PyTorch port
export WORLD_SIZE=$SLURM_NTASKS # Total processes
export RANK=$SLURM_PROCID       # Unique rank for each process

# Run the script inside the Singularity container
# Change Singularity image depending on environment
singularity exec --nv /gpfs/projects/ehpc176/Projects2/MambaMatting/matmat.sif \
    python /gpfs/scratch/ehpc176/Projects/Finetuning/mamba_train.py \
    --model-variant mamba \
    --dataset videomatte \
    --resolution-lr 512 \
    --seq-length-lr 15 \
    --learning-rate-backbone 0.0001 \
    --learning-rate-aspp 0.0002 \
    --learning-rate-decoder 0.0002 \
    --learning-rate-refiner 0 \
    --checkpoint-dir Experiments/Mamba/exp_00/stage1/checkpoint/ \
    --log-dir Experiments/Mamba/exp_00/stage1/tensorboard/ \
    --epoch-start 0 \
    --epoch-end 20

# sbatch -A ehpc327 -q acc_ehpc train.sh