#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=eval.py
#SBATCH --time=5:00:00
#SBATCH --partition=gpu
#SBATCH --account=kunf0097
#SBATCH --output=./outerr/%j.out
#SBATCH --error=./outerr/%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aminekidane.ghebreziabiher@ku.ac.ae
# SBATCH --nodelist=gpu-11-2
# SBATCH --exclusive
# SBATCH --mem=60000 
module load miniconda/3
conda activate llavanext
echo "Finally - out of queue" 
echo "Running on $(hostname)"
nvidia-smi

python -u /home/kunet.ae/ku5001069/LLaVA-NeXT/eval.py