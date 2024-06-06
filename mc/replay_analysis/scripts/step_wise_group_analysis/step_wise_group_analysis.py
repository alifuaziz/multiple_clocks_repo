"""
Run the functions for each subject in the a step wise manner where we save the results after each stage

After each stage is performed we run the next step

"""

# Get the subject list

# Python libraries
from pathlib import Path
from time import time

# Import custom scripts
import get_data_rdms
import evaluate_rsa
import get_vol_searchlights
# Create subject list from data folder 
# data_folder = Path("/Users/student/PycharmProjects/data")
DATA_FOLDER = Path("/home/fs0/chx061/scratch/data")
derivatives_folder = DATA_FOLDER / "derivatives"
subject_list = sorted([f for f in derivatives_folder.iterdir() if f.is_dir()])
# Remove group directory
subject_list = [x for x in subject_list if 'group' not in str(x)]
subject_list = [x for x in subject_list if 'sub-21' not in str(x)]


print(subject_list)


META_DATA = \
    {  
        'SUBJECT_DIRECTORY': str(DATA_FOLDER) + '/derivatives/sub-02',
        'SUB': 'sub-02',
        'EVS_TYPE': 'instruction_period',
        'RDM_VERSION': '01',
    }




for subject in subject_list:

    # UPDATE THE META_DATA for each subject
    META_DATA['SUBJECT_DIRECTORY'] = str(DATA_FOLDER) + '/derivatives/' + str(subject)[-6:] 
    META_DATA['SUB'] = str(subject)[-6:]

    print('Running script for subject: ', META_DATA['SUBJECT_DIRECTORY'])
    # Run the script
    get_vol_searchlights.main(META_DATA = META_DATA)

    pass


for subject in subject_list[20::]:

    # UPDATE THE META_DATA for each subject
    META_DATA['SUBJECT_DIRECTORY'] = str(DATA_FOLDER) + '/derivatives/' + str(subject)[-6:]
    META_DATA['SUB'] = str(subject)[-6:]

    print('Running script for subject: ', META_DATA['SUBJECT_DIRECTORY'])

    # Run the script
    get_data_rdms.main(META_DATA = META_DATA)

    pass


for subject in subject_list:
    # UPDATE THE META_DATA for each subject
    META_DATA['SUBJECT_DIRECTORY'] = str(DATA_FOLDER) + '/derivatives/' + str(subject)[-6:]
    META_DATA['SUB'] = str(subject)[-6:]

    print('Running script for subject: ', META_DATA['SUBJECT_DIRECTORY'])

    # Run the script
    evaluate_rsa.main(META_DATA = META_DATA)
    
    pass