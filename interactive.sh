#!/bin/bash

# Sbatch variables
PARTITION="shared-cpu"
NUM_NODES=1
NUM_TASKS=1
TOTAL_CPU_MEMORY=16gb
NUM_CPUS_PER_TASK=4
TIME_LIMIT=0-00:15:00

# Script variables
SIF_FOLDER=~/sif
SIF_NAME=mimic-triplets-image.sif  # for now, but should go for mimic_triplet-image.sif
SIF_IMAGE=${SIF_FOLDER}/${SIF_NAME}

# Start an interactive session with a shell in the Apptainer container
srun --job-name=mimic_triplet_interactive_shell \
     --partition=$PARTITION \
     --nodes=$NUM_NODES \
     --ntasks=$NUM_TASKS \
     --cpus-per-task=$NUM_CPUS_PER_TASK \
     --mem=$TOTAL_CPU_MEMORY \
     --time=$TIME_LIMIT \
     --pty apptainer shell ${SIF_IMAGE}