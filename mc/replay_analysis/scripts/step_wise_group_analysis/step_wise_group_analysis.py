"""
Run the functions for each subject in the a step wise manner where we save the results after each stage

After each stage is performed we run the next step

"""

# Get the subject list

# Python libraries
from pathlib import Path
from time import time

# Imoport custom scripts
from mc.replay_analysis.scripts import model_RDM_alif as model_RDM
from mc.replay_analysis.scripts import data_RDM_alif  as data_RDM

from mc.replay_analysis.scripts.step_wise_group_analysis import get_vol_searchlights
from mc.replay_analysis.scripts.step_wise_group_analysis import get_data_rdms
from mc.replay_analysis.scripts.step_wise_group_analysis import evaluate_rsa

# Create subject list from data folder 
data_folder = Path("/Users/student/PycharmProjects/data")
# data_folder = Path("/home/fs0/chx061/scratch/data")
derivatives_folder = data_folder / "derivatives"
subject_list = [f for f in derivatives_folder.iterdir() if f.is_dir()]
subject_list = subject_list[1::]

DATA_FOLDER = data_folder



META_DATA = \
    {  
        'SUBJECT_DIRECTORY': str(DATA_FOLDER) + '/derivatives/sub-02',
        'SUB': 'sub-02',
        'EVS_TYPE': 'instruction_period',
        'RDM_VERSION': '01',
    }




# for subject in subject_list:

#     # UPDATE THE META_DATA for each subject
#     META_DATA['SUBJECT_DIRECTORY'] = str(DATA_FOLDER) + '/derivatives/' + str(subject)[-6:] 
#     META_DATA['SUB'] = str(subject)[-6:]

#     print('Running script for subject: ', META_DATA['SUBJECT_DIRECTORY'])
#     # Run the script
#     get_vol_searchlights.main(META_DATA = META_DATA)

#     pass


for subject in subject_list:

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