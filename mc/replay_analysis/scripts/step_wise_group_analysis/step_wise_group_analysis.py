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

# Create subject list from data folder 
# data_folder = Path("/Users/student/PycharmProjects/data")
data_folder = Path("/home/fs0/chx061/scratch/data")
derivatives_folder = data_folder / "derivatives"
subject_list = [f for f in derivatives_folder.iterdir() if f.is_dir()]
subject_list = sorted(subject_list)
subject_list = subject_list[:20] + subject_list[21:]    # Remove the 20th element of the list

DATA_FOLDER = data_folder
# print(subject_list)


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


# for subject in subject_list:

#     # UPDATE THE META_DATA for each subject
#     META_DATA['SUBJECT_DIRECTORY'] = str(DATA_FOLDER) + '/derivatives/' + str(subject)[-6:]
#     META_DATA['SUB'] = str(subject)[-6:]

#     print('Running script for subject: ', META_DATA['SUBJECT_DIRECTORY'], "!!!!!!!")

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