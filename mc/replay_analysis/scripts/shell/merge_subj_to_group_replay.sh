# Set scratch directory for execution on server

scratchDir="/Users/student/PycharmProjects/data"
scratchDir="/home/fs0/chx061/scratch/data"

glm_version="01"
RSA_version="replay"


# Load FSL
module load fsl

# Defining the output group directory
groupDir=${scratchDir}/derivatives/group/group_RSA_${RSA_version}_glmbase_${glm_version}
echo This is group dir $groupDir
# If the directory does not exist, create it
if [ ! -d $groupDir ]; then
    echo "Group Directory does not exist. Creating it now."
    mkdir $groupDir
fi

# Get the example result directory
example_resultDir=${scratchDir}/derivatives/sub-02/results/results-standard-space

if [ ! -d $example_resultDir ]; then

    example_resultDir=${scratchDir}/derivatives/sub-02/results/results-standard-space
    # Search from the exaample_resultDir for all files that end with beta_std.nii.gz
    list_of_std_beta_files=$(find "$example_resultDir" -name "b_values_std.nii.gz" -type f)
else
    # Search from the exaample_resultDir for all files that end with beta_std.nii.gz
    list_of_std_beta_files=$(find "$example_resultDir" -name "b_values_std.nii.gz" -type f)
fi

# For each of the files in `example_resultDir`
echo This is example resultDir $example_resultDir
for file in $list_of_std_beta_files; do
    # Extract the filename
    filename=$(basename "$file")
    echo Now moving and merging $filename
    # No 01, 21 because this hasnt been processed yet
    for subjectTag in 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32 33 34 35; do
    # for subjectTag in 02 03; do
        # for every result file
        resultDir=${scratchDir}/derivatives/sub-${subjectTag}/results/results-standard-space
        # If the directory does not exist, try the other directory
        if [ ! -d $resultDir ]; then
            resultDir=${scratchDir}/derivatives/sub-${subjectTag}/results/results-standard-space
        fi

        echo Now for Subject $subjectTag and $resultDir
        # Check if the file exists in the result directory
        if [ ! -f ${groupDir}/${filename} ]; then
            cp ${resultDir}/$filename ${groupDir}/${filename}
        else 
            fslmerge -t ${groupDir}/${filename} ${resultDir}/$filename ${groupDir}/${filename}
        fi
    done
done
    
gunzip $(ls ${groupDir}/*.nii.gz)
echo done!
