"""
Run the functions for each subject in the a step wise manner where we save the results after each stage

After each stage is performed we run the next step

"""

# Get the subject list

# Python libraries
from pathlib import Path
from time import time

# Import custom scripts
import get_vol_searchlights 
import get_data_rdms
import evaluate_rsa
# Create subject list from data folder 
DATA_FOLDER = Path("/Users/student/PycharmProjects/data")
# DATA_FOLDER = Path("/home/fs0/chx061/scratch/data")
derivatives_folder = DATA_FOLDER / "derivatives"
subject_list = sorted([f for f in derivatives_folder.iterdir() if f.is_dir()])
# Remove group directory
subject_list = [x for x in subject_list if 'group' not in str(x)]
# Remove sub-21 since they were sleeping MRI machine 
subject_list = [x for x in subject_list if 'sub-21' not in str(x)]
# Start from sub-02
subject_list = subject_list[1:]

print(subject_list)


META_DATA = \
    {  
        'SUBJECT_DIRECTORY': str(DATA_FOLDER) + '/derivatives/sub-02',
        'SUB': 'sub-02',
        'EVS_TYPE': 'instruction_period',
        # 'RDM_VERSION': 'replay',
        'RDM_VERSION': 'replay_nan_off_diag',
        # 'RDM_VERSION': 'replay_zero_off_diag',
    }



# Create required directories for analysis
for subject in subject_list:
    # Create analysis type directory
    analysis_directory = Path(str(subject)) / 'analysis' / META_DATA['RDM_VERSION'] / 'preprocessing'
    if not analysis_directory.exists():
        analysis_directory.mkdir(parents=True)
    
    # Create results directory
    results_directory = Path(str(subject)) / 'analysis' / META_DATA['RDM_VERSION'] / 'results'
    if not results_directory.exists():
        results_directory.mkdir(parents=True)



# for subject in subject_list:

#     # UPDATE THE META_DATA for each subject
#     META_DATA['SUBJECT_DIRECTORY'] = str(DATA_FOLDER) + '/derivatives/' + str(subject)[-6:] 
#     META_DATA['SUB'] = str(subject)[-6:]

#     print('Running script for subject: ', META_DATA['SUBJECT_DIRECTORY'])
#     # Run the script
#     get_vol_searchlights.main(META_DATA = META_DATA)

#     pass


# for subject in subject_list:

#     # UPDATE THE META_DATA for each subject
#     META_DATA['SUBJECT_DIRECTORY'] = str(DATA_FOLDER) + '/derivatives/' + str(subject)[-6:]
#     META_DATA['SUB'] = str(subject)[-6:]

#     print('Running script for subject: ', META_DATA['SUBJECT_DIRECTORY'])

#     # Run the script
#     get_data_rdms.main(META_DATA = META_DATA)

#     pass


for subject in subject_list:
    # UPDATE THE META_DATA for each subject
    META_DATA['SUBJECT_DIRECTORY'] = str(DATA_FOLDER) + '/derivatives/' + str(subject)[-6:]
    META_DATA['SUB'] = str(subject)[-6:]

    print('Running script for subject: ', META_DATA['SUBJECT_DIRECTORY'])

    # Run the script
    evaluate_rsa.main(META_DATA = META_DATA)
    
    pass