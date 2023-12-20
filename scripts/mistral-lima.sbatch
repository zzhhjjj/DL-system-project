#!/bin/bash
#SBATCH --job-name=lima2500
#SBATCH --output=output_lima2500.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --mem=30GB
#SBATCH --mail-type=END
#SBATCH --mail-user=hz3308@nyu.edu
#SBATCH --output=lima-from-2500%j.out

module purge

singularity exec --nv --overlay \
/scratch/hz3308/pytorch-example/overlay-25GB-500K.ext3:ro \
/scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
/bin/bash -c 'source /ext3/env.sh; cd /scratch/hz3308/notebooks;\
conda activate chatpdf; \
python mistral-lima-finetune.py'