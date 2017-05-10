#!/bin/sh
#SBATCH --job-name="tensorflow"
# SBATCH --array=1
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=1
#SBATCH -N 6
#SBATCH --cpus-per-task=23
#SBATCH --partition=kraken_fast
#SBATCH --mem=MaxMemPerNode
#SBATCH -t 0-10:00:59
#SBATCH --output=arrayJob_%A_%a.out
#SBATCH --error=arrayJob_%A_%a.err

# ...set variable for working directory
workingDir=/home/slurm/ackdav-${SLURM_JOB_ID}

cd ~/tf-sparql/graph_vec_nn/

/home/user/ackdav/srun --ntasks $SLURM_NTASKS --ntasks-per-node=$SLURM_NTASKS_PER_NODE -c $SLURM_CPUS_PER_TASK time python rnn-distributed.py
