#!/bin/bash

#SBATCH --job-name=singularity_build
#SBATCH --partition=shared-gpu,private-teodoro-gpu
#SBATCH --nodelist=gpu023,gpu024,gpu036,gpu037,gpu038,gpu039,gpu040,gpu041,gpu042,gpu043
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=./logs/singularity_build_%j.txt
#SBATCH --error=./logs/singularity_build_%j.err

# Check if the .sif file exists, and only remove it if it does
mkdir -p ~/sif
if [ -f ~/sif/mimic-triplets-image.sif ]; then
    rm ~/sif/mimic-triplets-image.sif
fi

# Build the Singularity image
apptainer build ~/sif/mimic-triplets-image.sif mimic-triplets-image.def
