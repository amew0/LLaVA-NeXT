#!/bin/bash
#SBATCH -N 1
#SBATCH -c 26
#SBATCH -J ft-s2-7b.sh
#SBATCH -t 24:00:00
#SBATCH -p gpu
#SBATCH -A kunf0097
#SBATCH -o ./outerr/%j.out
#SBATCH -e ./outerr/%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aminekidane4@gmail.com
#SBATCH --gres=gpu:4
# SBATCH --nodelist=gpu-10-2
module load miniconda
conda activate llavanext
echo "Finally - out of queue" 
echo "Running on $(hostname)"
nvidia-smi

/home/kunet.ae/ku5001069/LLaVA-NeXT/scripts/train/ft_s2_7b.sh