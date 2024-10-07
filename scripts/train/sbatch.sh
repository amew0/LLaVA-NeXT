#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=finetune_ov_wzno_gpt_context
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --account=kunf0097
#SBATCH --output=./%j.out
#SBATCH --error=./%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aminekidane.ghebreziabiher@ku.ac.ae
#SBATCH --nodelist=gpu-11-2
# SBATCH --exclusive
# SBATCH --mem=60000 
module load miniconda/3
conda activate llavanext
echo "Finally - out of queue" 
nvidia-smi

/home/kunet.ae/ku5001069/LLaVA-NeXT/scripts/train/finetune_ov_wzno_gpt_context.sh
