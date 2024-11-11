#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --job-name=direct-s2.sh
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu
#SBATCH --account=kunf0097
#SBATCH --output=./outerr/%j.out
#SBATCH --error=./outerr/%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aminekidane4@gmail.com
# SBATCH --exclusive
# SBATCH --nodelist=gpu-11-2
module load miniconda/3
conda activate llavanext
echo "Finally - out of queue" 
echo "Running on $(hostname)"
nvidia-smi

/home/kunet.ae/ku5001069/LLaVA-NeXT/scripts/train/ft_s2.sh