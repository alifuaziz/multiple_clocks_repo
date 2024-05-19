"""
Files for running the group replay analysis.

Order of operations. 

1. Create a list of all the subjects in the data folder. 

2. Run the script to create the model RDM for each subject and save it to the appropriate model RDM folder

3. Run script to create the data RDM for the searchlights for the subjects and they compute the correlation between the model and data RDMs
"""
# Python libraries
from pathlib import Path
from time import time

# Imoport custom scripts
from mc.replay_analysis.scripts import alif_RSA_analysis
from mc.replay_analysis.scripts import alif_RSA_model_RDM



# Create subject list from data folder 
data_folder = Path("/Users/student/PycharmProjects/data")
derivatives_folder = data_folder / "derivatives"
subject_list = [f for f in derivatives_folder.iterdir() if f.is_dir()]
# subject_list = subject_list[1::]





# Run the analysis for each subject
for subject in subject_list:

    str_subject = str(subject)[-6:]
    print("Running Analysis for Subject: ", str_subject)
    model_rdm_analysis_settings = \
    {
        "SUBJECT_NO": str_subject,
        "REGRESSION_VERSION": "01",
        "RDM_VERSION": "01",
        "DATA_DIR": data_folder,
        "TEMPORAL_RESOLUTION": 10,
        "MODEL": "replay-2",
        "RDM_SIMILARITY_MEASURE": "pearson",
        "RDM_VISUALISE": False,
        "FMRI_PLOTTING": False,
        "FMRI_SAVE": False
    }



    data_rdm_analysis_settings = \
    {
        'SUBJECT_NO': str_subject,
        'REGRESSION_VERSION': "01",
        'RDM_VERSION': "01",
        'DATA_DIR': data_folder,
        'EVS_TYPE': "instruction_period",
        'REMOVE_AUTOCORR': True, 
        'USE_PREVIOUS_SEARCHLIGHTS': False,
        'SEACHLIGHT_RADIUS': 3,
        'SEACHLIGHT_THRESHOLD': 0.5,
        'TASK_HALVES': ['1', '2'],
        'USE_PREVIOUS_DATA_RDM': False,
        'VISUALISE_RDMS': False,

    }

    print("Running Analysis for Subject: ", subject)
    print("Creating Model RDM for Subject: ", subject)
    # Run script to create model RDM for subject
    alif_RSA_analysis.model_RDM_script(model_rdm_analysis_settings = model_rdm_analysis_settings)


    print("Creating Data RDM for Subject: ", subject)
    alif_RSA_model_RDM.data_RDM_script(data_rdm_analysis_settings = data_rdm_analysis_settings)
    # Run script to create data RDM for subject




