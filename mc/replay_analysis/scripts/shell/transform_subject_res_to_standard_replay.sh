#!/bin/sh
# transforms beta-results per model and subject to standard space
# to prepare group stats.
# submit like bash transform_subject_res_to_standard.sh
# requires results from submit_RSA_fmri.sh


# Load FSL
module load fsl

# Set scratch directory for execution on server
scratchDir="/home/fs0/chx061/scratch/data"

analysisType="replay"


echo Now transforming results from replay analysis to standard space

# MISSING 21 because this hasnt been processed yet
for subjectTag in 02; do
# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32 33 34 35; do
    # Get the directory of the results
    resultDir=${scratchDir}/derivatives/sub-${subjectTag}/analysis/replay/results

    # Get the standard space directory
    stdDir=${scratchDir}/derivatives/sub-${subjectTag}/analysis/replay/results-standard-space

    # if the direcotry exists, delete all files in it
    if [ -d $stdDir ]; then
        echo The standard space directory already exists. Deleting all files in this directory: $stdDir
        rm ${stdDir}/*_std.nii.gz
    fi
    # if the directory does not exist, create it
    if [ ! -d $stdDir ]; then
        echo The standard space directory does not exist. Creating this directory to save standard files: $stdDir
        mkdir $stdDir
    fi

    # Get the preprocessed directory, where the standard image is located
    preprocDir=${scratchDir}/derivatives/sub-${subjectTag}/func/preproc_clean_01.feat

    # Removing previous standard space files to create the new ones
    echo Running FLIRT for Subject $subjectTag
    # in case something has gone wrong before
    find "$resultDir" -type f -name 'std-std-*.nii.gz' -exec rm {} +
    find "$resultDir" -type f -name 'std-*.nii.gz' -exec rm {} +
    find "$resultDir" -type f -name '*_std.nii.gz' -exec rm {} +

    # Loop through each .nii.gz file in the directory
    for file in "$resultDir"/*.nii.gz; do
        
        # Extract the filename without the extension
        file_name=$(basename "$file" .nii.gz)

        # Skip the processing of this file if it has already been transformed
        if [[ $file_name == std* ]]; then
            continue 
        fi
        
        # Define the output filename
        output="${stdDir}/${file_name}_std.nii.gz"
        # Transform to standard space
        flirt -in "$file" -ref ${preprocDir}/reg/standard.nii.gz -applyxfm -init ${preprocDir}/reg/example_func2standard.mat -out "$output"
    done
done


# Explanation of FLIRT commmand arguments:
# - in : input file that is transformed to standard space 
# - ref : reference volume. An affine matrix is calcualted that registers the input to the reference 
# - applyxfm : apply the affine matrix to the input file
# - init : initial affine matrix that is applied to the input file
# - out : output file that is saved in the standard space directory