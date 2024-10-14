#!/bin/bash

# Common sbatch variables
PARTITION="shared-cpu"
NUM_NODES=1
NUM_TASKS=1
TIME_LIMIT="0-00:15:00"
TOTAL_CPU_MEMORY=64gb
NUM_CPUS_PER_TASK=32

# Script variables
SIF_FOLDER=~/sif
SIF_NAME=mimic-triplets-image.sif
SIF_IMAGE=${SIF_FOLDER}/${SIF_NAME}
SCRIPT=data_utils.py

# Build launch the script in apptainer container
sbatch --job-name=data_utils \
       --partition=$PARTITION \
       --nodes=$NUM_NODES \
       --ntasks=$NUM_TASKS \
       --cpus-per-task=$NUM_CPUS_PER_TASK \
       --mem=$TOTAL_CPU_MEMORY \
       --time=$TIME_LIMIT \
       --output=./results/logs/job_%j.txt \
       --error=./results/logs/job_%j.err \
       --wrap="srun apptainer exec ${SIF_IMAGE} python ${SCRIPT}"
