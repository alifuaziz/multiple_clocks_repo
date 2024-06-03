#!/bin/sh
# Run flirt for all participants
# Submit this script to the cluster to run the flirt

# Scratch Directory
scratchDir="/Users/student/PycharmProjects/data"
# scratchDir="/home/fs0/chx061/scratch/data"

# Load the fsl module
# module load fsl

# Get this list of participants in the derivatives directory
list_of_participants=$(ls ${scratchDir}/derivatives/)

echo $list_of_participants


# # For each participant 
for participant in $list_of_participants; do
    # get the directory for the participant
    participantDir=${scratchDir}/derivatives/${participant}
    echo $participantDir