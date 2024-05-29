# Python libraries
from pathlib import Path
from time import time

from mc.replay_analysis.scripts import script_data_rdm as data_rdms

# Create subject list from data folder 
data_folder = Path("/Users/student/PycharmProjects/data")
derivatives_folder = data_folder / "derivatives"
subject_list = [f for f in derivatives_folder.iterdir() if f.is_dir()]
# subject_list = subject_list[1::]


#%%

print("test")

#%%
# Run the analysis for each subject
for subject in subject_list:

    analysis_options = \
    {
        'SUB': str(subject)[-6:],
        'EVS_TYPE': 'instruction_period',
        'MODEL_TYPE': 'replay',
        'RDM_SIZE': 'cross_corr',
        'RDM_VERSION': '01',
        'SUBJECT_DIRECTORY': data_folder / "derivatives" / str(subject),
        'RESULTS_DIRECTORY': data_folder / "derivatives" / str(subject) / "func" / "RSA_replay"
    }


    data_rdms.replay_analysis(
        analysis_options = analysis_options
    )
