#!/bin/bash
#SBATCH -N 1
#SBATCH -c 20
#SBATCH -J ft-s2.sh
#SBATCH -t 24:00:00
#SBATCH -p gpu
#SBATCH -A kunf0097
#SBATCH -o ./outerr/%j.out
#SBATCH -e ./outerr/%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aminekidane4@gmail.com
# SBATCH --begin=02:00:00
# SBATCH --exclusive
# SBATCH --nodelist=gpu-10-3
# SBATCH --mem-per-cpu=32000
module load miniconda/3
conda activate llavanext
echo "Finally - out of queue" 
echo "Running on $(hostname)"
nvidia-smi

/home/kunet.ae/ku5001069/LLaVA-NeXT/scripts/train/ft_s2.sh