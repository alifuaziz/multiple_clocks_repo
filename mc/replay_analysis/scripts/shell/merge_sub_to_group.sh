# !/bin/bash
# Set scratch directory for execution on server
# scratchDir="/Users/student/PycharmProjects/data"
scratchDir="/vols/Scratch/chx061/data"

# 
glm_version="instruction_period"
RSA_version="replay"

# Load FSL
module load fsl

# Defining the output group directory
groupDir=${scratchDir}/derivatives/group/group_RSA_${RSA_version}_glmbase_${glm_version}

# If the directory does not exist, create it
if [ ! -d $groupDir ]; then
    mkdir $groupDir
fi

# Get the example result directory
example_resultDir=${scratchDir}/derivatives/sub-03/func/replay/results-standard-space

list_of_std_b_files=$(find "$example_resultDir" -name "b_values_std.nii.gz" -type f)

# example_resultDir=${scratchDir}/derivatives/sub-02/func/RSA_${RSA_version}_glmbase_${glm_version}/results-standard-space

# if [ ! -d $example_resultDir ]; then
#     example_resultDir=${scratchDir}/derivatives/sub-02/func/RSA_${RSA_version}_${glm_version}/results-standard-space
#     list_of_std_beta_files=$(find "$example_resultDir" -name "avg*beta_std.nii.gz" -type f)
# else
#     list_of_std_beta_files=$(find "$example_resultDir" -name "*beta_std.nii.gz" -type f)
# fi

# remove all the stuff from the directoty 

echo this is example resultDir $example_resultDir
# Then, for each of these files
for file in $list_of_std_b_files; do
    # Extract the filename
    filename=$(basename "$file")
    echo Now moving and merging $filename from the list of b files
    # no 01, 21
    # for subjectTag in 02 03; do
    for subjectTag in 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32 33 34 35; do
        
        # Get the results directory for the subject
        resultDir=${scratchDir}/derivatives/sub-${subjectTag}/func/replay/results-standard-space
        # Create the directory is it does not exist
        if [ ! -d $resultDir ]; then
            resultDir=${scratchDir}/derivatives/sub-${subjectTag}/func/replay/results-standard-space
        fi

        echo For each subject $subjectTag and result directory $resultDir
        # Check if the file for each subject exists in the group directory
        if [ ! -f ${groupDir}/${filename} ]; then
            # If is does not, then copy the file to the group directory
            cp ${resultDir}/$filename ${groupDir}/${filename}
        else 
            echo Output file ${groupDir}/${filename} 
            echo Image 1 ${resultDir}/$filename
            echo Image 2 ${groupDir}/${filename}
            # If it does, then merge the file with the existing file in the group directory
            fslmerge -t ${groupDir}/${filename} ${resultDir}/$filename ${groupDir}/${filename}
        fi
    done
done
    
# gunzip $(ls ${groupDir}/*.nii.gz)
echo done!


# Useage 
# fslmerge -t output_file input_file1 input_file2
# -t: merge in time
# output_file: the output file
# input_file1: the first input file
# input_file2: the second input file