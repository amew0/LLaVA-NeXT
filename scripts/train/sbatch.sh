#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --job-name=ft-s2.sh
#SBATCH --time=3:00:00
#SBATCH --partition=gpu
#SBATCH --account=kunf0097
#SBATCH --output=./outerr/%j.out
#SBATCH --error=./outerr/%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aminekidane.ghebreziabiher@ku.ac.ae
# SBATCH --exclusive
module load miniconda/3
conda activate llavanext
echo "Finally - out of queue" 
echo "Running on $(hostname)"
nvidia-smi

# /home/kunet.ae/ku5001069/LLaVA-NeXT/scripts/train/ft_s3.sh
/home/kunet.ae/ku5001069/LLaVA-NeXT/scripts/train/finetune_ov_wzno_gpt_context.sh