#!/bin/sh
# Run PALM for group stats
# Svenja Kuchenhoff 2024
# run like bash Group_PALM.sh
# important: run module load palm first!

# Set scratch directory for execution on server
# scratchDir="/Users/student/PycharmProjects/data"
# fslDir="/Users/student/fsl"

scratchDir="/home/fs0/chx061/scratch/data"
# fslDir="/opt/fmrib/fsl"

glm_version="instruction_period"
# RSA_version="replay"
# RSA_version="difficulty"
RSA_version="replay_nan_off_diag"
# RSA_version="replay_zero_off_diag"


module load PALM/032024-MATLAB-2024a
# module load fsl

# needs to be unzipped files!
groupDir=${scratchDir}/derivatives/group/group_RSA_${RSA_version}_glmbase_${glm_version}
# Check if the directory exists
if [ ! -d "$groupDir" ]; then
    echo "Group Directory does not exist."
    exit 1
else
    echo Folder with concatenated files for permutation testing: $groupDir
fi

# HYPERPARAMETERS FOR PALM
# Both mask and input files should be unzipped (.nii)
clusterThreshold=3.1
permutationNumber=1000 #should be something like 1000 or 5000 later..
# maskFile=${scratchDir}/masks/MNI152_T1_2mm_brain_mask.nii.gz


# Construct the folder for permutation testing for this analysis
permDir=$scratchDir/derivatives/group/RSA_${RSA_version}_glmbase_${glm_version}_PALM
if [ ! -d "$permDir" ]; then
    mkdir ${permDir}
fi




echo Group Directory $groupDir

# Loop through all files in the RSA group results directory
for curr_file in "$groupDir"/*; do
    # Check if it's a regular file
    # NOTE:what should the file format for this be? It should be the concatentated file from the output of the fslmerge script. This is the file we are doing the permutation testing on.
    echo "Current file: $curr_file"


    if [ -f "$curr_file" ]; then
        # Set path for output file
        old_file_name=$(basename "${curr_file}")
        echo "File basename: $old_file_name"
        # remove extension of the file name
        file_name="${old_file_name%.*}"
        echo "File name without extension: $file_name"
        # Set the output path for the file
        outPath=$permDir/${file_name}
        echo "Output filename of PALM: $outPath"
        # Run PALM
        palm -i ${curr_file} -T -C $clusterThreshold -Cstat mass -n $permutationNumber -o $outPath -ise -save1-p
        # echo "Processed: $curr_file"
    fi
done


echo "Done with PALM permutation testing for all files in the group directory"

# PALM Usage and documenation  https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/PALM/UserGuide
# -i <file>: Input(s). More than one can be specified, each one preceded by its own -i. All input files must contain the same number of observations (e.g., the same number of subjects). Except for NPC and MV, mixing is allowed (e.g., voxelwise, vertexwise and non-imaging data can be all loaded at once, and later will be all corrected across). 
# -T: Enable TFCE inference for univariate (partial) tests, as well as for NPC and/or MV if these options have been enabled. 
# -C <real>: Enable cluster inference for univariate (partial) tests, with the supplied cluster-forming threshold (supplied as the equivalent z-score), as well as for NPC and/or MV if these options have been enabled. 
# -Cstat <name>: Choose which cluster statistic should be used. Accepted statistics are "extent" and "mass". 
# -n <integer>: Number of permutations to perform.
# -o <string>: Output directory.
# -ise: Assume independent and symmetric errors (ISE), to allow sign-flipping (more details below).
# -save1-p: Save (1-p) instead of the actual p-values. Instead of -save1-p, consider using -logp. 