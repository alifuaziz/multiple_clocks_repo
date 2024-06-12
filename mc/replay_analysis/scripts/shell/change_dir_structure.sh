# !/bin/bash
# # Set scratch directory for execution on server
scratchDir="/vols/Scratch/chx061/data"
scratchDir="/home/fs0/psy"
# Get a list of all the directories in the derivatives directory
derivativesDir=${scratchDir}/derivatives

# Get the list of all the subjects and remove the group directory
list_of_subjects=$(ls $derivativesDir | grep -v group)

# Loop through each subject
for subject in $list_of_subjects; do
    echo $subject
    # analysisDir=${scratchDir}/derivatives/${subject}/analysis
    # if [ ! -d $analysisDir ]; then
    #     mkdir $analysisDir
    # fi

    # # move replay directory to analysis directory
    # replayDir=${scratchDir}/derivatives/${subject}/replay
    # if [ -d $replayDir ]; then
    #     mv $replayDir $analysisDir
    # fi
    
    # # delete the results directory 
    # resultsDir=${scratchDir}/derivatives/${subject}/results
    # if [ -d $resultsDir ]; then
    #     rm -r $resultsDir
    # fi

    # get a list of files that are not directories from subject directory
    # list_of_files=$(ls $derivativesDir/${subject} | grep -v analysis | grep -v anat | grep -v func | grep -v beh)
    # echo $list_of_files

    # # for each file in the list of files
    # for file in $list_of_files; do
    #     # get the file path
    #     filePath=${derivativesDir}/${subject}/${file}

    #     preprocessingDir=${derivativesDir}/${subject}/analysis/preprocessing
    #     if [ ! -d $preprocessingDir ]; then
    #         mkdir $preprocessingDir
    #     fi

    #     # get the new file path
    #     newFilePath=${derivativesDir}/${subject}/analysis/preprocessing/${file}
    #     # move the file to the new file path
    #     mv $filePath $newFilePath
    # done

# end for loop
done