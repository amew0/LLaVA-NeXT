#!/bin/bash
#SBATCH -n 1
#SBATCH -c 48
#SBATCH --job-name=3d
#SBATCH -t 3-00:00:00
#SBATCH -p gpu
#SBATCH -A kunf0097
#SBATCH --output=./outerr/%j.out
#SBATCH --error=./outerr/%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aminekidane4@gmail.com
# SBATCH --exclusive
#SBATCH --nodelist=gpu-11-3
# module load miniconda/3
# conda activate llavanext
echo "Finally - out of queue" 
echo "Running on $(hostname)"
nvidia-smi

# /home/kunet.ae/ku5001069/LLaVA-NeXT/scripts/train/ft_s2.sh
python -c "import time; time.sleep(3*86400)"
