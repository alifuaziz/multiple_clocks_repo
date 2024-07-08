#!/bin/sh

# Define the group directory
groupDir='/home/fs0/chx061/scratch/data/derivatives/group/group_RSA_replay_glmbase_instruction_period_sliding_window'

# Get all the folders with the name TR in the group directory
list_of_folders=$(ls $groupDir | grep TR)

# for each folder in the group directory get the compressed file and uncompress it in the same directory it is located
for folder in $list_of_folders; do
    echo $folder
    # Get the compressed file
    compressed_file=$(ls $groupDir/$folder/*.nii.gz)
    # Uncompress the file
    echo compressed_file $compressed_file
    gunzip -f $compressed_file 

done