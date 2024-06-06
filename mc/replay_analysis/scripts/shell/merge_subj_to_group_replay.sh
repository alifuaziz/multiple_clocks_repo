# Set scratch directory for execution on server
scratchDir= "/home/fs0/chx061/scratch/data"
analysisDir="/home/fs0/chx061/scratch/analysis"

glm_version="01"
RSA_version="replay"


groupDir=${scratchDir}/derivatives/group/group_RSA_${RSA_version}_glmbase_${glm_version}
echo this is group dir $groupDir
if [ ! -d $groupDir ]; then
    mkdir $groupDir
fi

example_resultDir=${scratchDir}/derivatives/sub-02/func/RSA_${RSA_version}_glmbase_${glm_version}/results-standard-space

if [ ! -d $example_resultDir ]; then
    example_resultDir=${scratchDir}/derivatives/sub-02/func/RSA_${RSA_version}_${glm_version}/results-standard-space
    list_of_std_beta_files=$(find "$example_resultDir" -name "avg*beta_std.nii.gz" -type f)
else
    list_of_std_beta_files=$(find "$example_resultDir" -name "*beta_std.nii.gz" -type f)
fi

echo this is example resultDir $example_resultDir
# Then, for each of these files
for file in $list_of_std_beta_files; do
    # Extract the filename
    filename=$(basename "$file")
    echo now moving and merging $filename
    # no 21 
    # also find a way to include sub 33 and 34 in glm 06 rsa 05 once they are done!!
    for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32 33; do
        # for every result file
        resultDir=${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}_glmbase_${glm_version}/results-standard-space
        if [ ! -d $resultDir ]; then
            resultDir=${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}_${glm_version}/results-standard-space
        fi

        echo now for subject $subjectTag and $resultDir
        if [ ! -f ${groupDir}/${filename} ]; then
            cp ${resultDir}/$filename ${groupDir}/${filename}
        else 
            fslmerge -t ${groupDir}/${filename} ${resultDir}/$filename ${groupDir}/${filename}
        fi
    done
done
    
gunzip $(ls ${groupDir}/*.nii.gz)
echo done!

