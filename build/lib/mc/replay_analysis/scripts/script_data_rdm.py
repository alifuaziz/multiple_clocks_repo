# %% 
# Import 
import nibabel as nib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import seaborn as sns 
from joblib import Parallel, delayed
from pathlib import Path
from tqdm import tqdm
import pickle
sns.set_style('dark')
# from rsatoolbox.util.searchlight import get_volume_searchlight
from nilearn.image import load_img


# Imports
import mc.replay_analysis.functions.model_rdms as model_rdms
import mc.replay_analysis.functions.data_rdms as data_rdms
import mc.analyse.analyse_MRI_behav as analyse_MRI_behav
from mc.analyse.searchlight import get_volume_searchlight

# %% Define the Subject data and the EVs that are being used to create the RDMs
SUB = "sub-03"
EVS_TYPE = 'instruction_period'
MODEL_TYPE = 'replay'
RDM_SIZE = "cross_corr"
TASK_HALVES = ['1', '2']
RDM_VERSION = '01'

# LOADING IN PICK FILES 
LOAD_VOLU_SEARCHLIGHT = False
LOAD_DATA_SEARCHLIGHT = False
LOAD_DATA_RDM = False



SUBJECT_DIRECTORY = Path('/home/fs0/chx061/scratch/data/derivatives/' + SUB + '/')
RESULTS_DIRECTORY = Path('/home/fs0/chx061/scratch/data/derivatives/' + SUB + '/func/RSA_replay/')
# Create the directories if they do not exist
if not RESULTS_DIRECTORY.exists():
    RESULTS_DIRECTORY.mkdir(parents=True)
if not SUBJECT_DIRECTORY.exists():
    SUBJECT_DIRECTORY.mkdir(parents=True)



# %%


def replay_analysis(**analysis_options):

# %%
    SUB         = analysis_options['analysis_options'].get('SUB', 'sub-02')
    EVS_TYPE    = analysis_options['analysis_options'].get('EVS_TYPE', 'instruction_period')
    MODEL_TYPE  = analysis_options['analysis_options'].get('MODEL_TYPE', 'replay')
    RDM_SIZE    = analysis_options['analysis_options'].get('RDM_SIZE', 'cross_corr')
# %%
    SUB         = analysis_options['analysis_options'].get('SUB', 'sub-02')
    EVS_TYPE    = analysis_options['analysis_options'].get('EVS_TYPE', 'instruction_period')
    MODEL_TYPE  = analysis_options['analysis_options'].get('MODEL_TYPE', 'replay')
    RDM_SIZE    = analysis_options['analysis_options'].get('RDM_SIZE', 'cross_corr')
    # TASK_HALVES = analysis_options.get('TASK_HALVES', ['1', '2'])
    RDM_VERSION = analysis_options['analysis_options'].get('RDM_VERSION', '01')
    RDM_VERSION = analysis_options['analysis_options'].get('RDM_VERSION', '01')


    # Directory for the data and results
    SUBJECT_DIRECTORY = analysis_options['analysis_options'].get('SUBJECT_DIRECTORY', Path('/Users/student/PycharmProjects/data/derivatives/' + SUB + '/'))
    RESULTS_DIRECTORY = analysis_options['analysis_options'].get('RESULTS_DIRECTORY', Path('/Users/student/PycharmProjects/data/derivatives/' + SUB + '/func/RSA_replay/'))
    SUBJECT_DIRECTORY = analysis_options['analysis_options'].get('SUBJECT_DIRECTORY', Path('/Users/student/PycharmProjects/data/derivatives/' + SUB + '/'))
    RESULTS_DIRECTORY = analysis_options['analysis_options'].get('RESULTS_DIRECTORY', Path('/Users/student/PycharmProjects/data/derivatives/' + SUB + '/func/RSA_replay/'))


    print(SUBJECT_DIRECTORY)
    print(SUBJECT_DIRECTORY)
    # Create the directories if they do not exist
    if not RESULTS_DIRECTORY.exists():
        RESULTS_DIRECTORY.mkdir(parents=True)
    if not SUBJECT_DIRECTORY.exists():
        SUBJECT_DIRECTORY.mkdir(parents=True)





    #%% Load in the EVs for the instruction from the correct direcory

    # Create dictionary to store the data for each EV for both task halves
    # Convert the output of this to be pandas arrays

    EVs_both_halves_dict = {
        '1': None,
        '2': None
    }
    # create new dictionary to store the 2D array of EVs for both task halves
    EVs_both_halves_2d = EVs_both_halves_dict.copy()

        
    EVs_path_dict = data_rdms.get_EV_path_dict(
        subject_directory = SUBJECT_DIRECTORY,
        EVs_type = EVS_TYPE
        )

    # Load in the EVs for the instruction periods from the dictionary of paths
    EVs_data_dict = data_rdms.load_EV_data(
        EVs_path_dict = EVs_path_dict,
        RDM_VERSION = RDM_VERSION
    )

    # Unravel the values of each EV
    EVs_data_dict = data_rdms.unravel_EV_data(EVs_data_dict)
    EVs_data_dict = pd.DataFrame(EVs_data_dict)


    # get column names
    column_names = EVs_data_dict.columns.tolist()
    for idx, name in enumerate(column_names):
        column_names[idx] = name[3:7]
        

    # column_names
    EVs_data_dict.rename(columns=dict(zip(EVs_data_dict.columns.tolist(), column_names)), inplace=True)

    # Get the list of conditions in a standard order
    conditions_std_order = data_rdms.get_standard_order()
    # reorder the columns 
    EVs_data_dict.loc[:, conditions_std_order]


    # Load binary mask of the brain
    mask = load_img(f"{SUBJECT_DIRECTORY}/anat/{SUB}_T1w_noCSF_brain_mask_bin_func_01.nii.gz")
    mask_array = mask.get_fdata()
    # Get list of voxel centers and their volume neighbours
    centers, vol_neighbors = get_volume_searchlight(
        mask = mask_array,
        radius = 3,
        threshold = 0.5)

    # Deals with searchlights that are not the correct size
    # Deals with searchlights that are not the correct size
    vol_neighbors = data_rdms.resize_neighbors(
        vol_neighbors = vol_neighbors,
        size = 93
    )

    #%% Turns the vol_neighbours into a df of vol_searchlights
    # Create a data frame to store the searchliht data inside of

    vol_searchlight = data_rdms.create_vol_searchlight_dataframe(
        vol_neighbors = vol_neighbors,
        centers = centers
    )

    # %% Save the vol_searchlight to a pickle file
    with open(f"{SUBJECT_DIRECTORY}/vol_searchlight_df.pkl", 'wb') as f:
        pickle.dump(vol_searchlight, f)

    # load vol_searchlight from a pickle file
    # with open(f"{SUBJECT_DIRECTORY}/vol_searchlight_df.pkl", 'rb') as f:
    #     vol_searchlight = pickle.load(f)


    # %% Save the vol_searchlight to a pickle file
    with open(f"{SUBJECT_DIRECTORY}/vol_searchlight_df.pkl", 'wb') as f:
        pickle.dump(vol_searchlight, f)

    # load vol_searchlight from a pickle file
    # with open(f"{SUBJECT_DIRECTORY}/vol_searchlight_df.pkl", 'rb') as f:
    #     vol_searchlight = pickle.load(f)


    #%% Create the data searchlights from the vol_searchlights

    data_searchlight = data_rdms.get_data_searchlight(
        vol_searchlight = vol_searchlight,
        EVs_data_dict = EVs_data_dict,
    )
    #%% Save the data searchlights because they take ages to calculate

    # save data_searchlight to a pickle file
    # with open(f"{SUBJECT_DIRECTORY}/searchlight_data_searchlight.pkl", 'wb') as f:
    #     pickle.dump(data_searchlight, f)
    # with open(f"{SUBJECT_DIRECTORY}/searchlight_data_searchlight.pkl", 'wb') as f:
    #     pickle.dump(data_searchlight, f)

    # load data_searchlight from a pickle file
    with open(f"{SUBJECT_DIRECTORY}/searchlight_data_searchlight.pkl", 'rb') as f:
        data_searchlight = pickle.load(f)
    with open(f"{SUBJECT_DIRECTORY}/searchlight_data_searchlight.pkl", 'rb') as f:
        data_searchlight = pickle.load(f)

    #%% Get the RDMs for each searchlight

    data_rdms_dict = data_rdms.get_data_rdms(
        data_searchlight = data_searchlight,
        )

    #%% Save the data RDMs to a pickle file

    # save data_rdms_dict to a pickle file
    with open(f"{SUBJECT_DIRECTORY}/searchlight_data_rdms.pkl", 'wb') as f:
        pickle.dump(data_rdms_dict, f)

    # load data_rdms_dict from a pickle file
    # with open(f"{SUBJECT_DIRECTORY}/searchlight_data_rdms.pkl", 'rb') as f:
    #     data_rdms_dict = pickle.load(f)


    #%% Save the data RDMs to a pickle file

    # save data_rdms_dict to a pickle file
    with open(f"{SUBJECT_DIRECTORY}/searchlight_data_rdms.pkl", 'wb') as f:
        pickle.dump(data_rdms_dict, f)

    # load data_rdms_dict from a pickle file
    # with open(f"{SUBJECT_DIRECTORY}/searchlight_data_rdms.pkl", 'rb') as f:
    #     data_rdms_dict = pickle.load(f)


    #%% Convert the Data RDMs to upper triangle vectors for RSA 
    # data_rdms_tri = data_rdms.get_data_rdms_tri(
    #     data_rdms_dict = data_rdms_dict
    #     )


    #%% Load the model RDMs for the replay model

    # RETURNING THE INCORRECT DICTIONARY. GETTING AN ERROR
    model_rdms_dict = model_rdms.get_model_rdms(
        conditions = conditions_std_order, 
        TYPE = MODEL_TYPE, 
        SIZE = RDM_SIZE
        )

    # Convert the model RDMs to upper triangle vectors for RSA
    # model_rdms_dict_tri = data_rdms.get_data_rdms_tri(
    #     model_rdms_dict
    #     )

    #%% Evalute the correlation between the model and data RDMs

    eval_result = Parallel(n_jobs=-1)(
        delayed(data_rdms.evaluate_model)(
            Y=data_rdms_tri[center],
            X=model_rdms_dict_tri[MODEL_TYPE]
        ) for center in tqdm(data_rdms_tri.columns, desc="Data Searchlights Running")
    )

    #%% Save the data RDMs .nii maps of the parameters 

    data_rdms.save_RSA_result(
        results_file = eval_result,
        data_rdms_tri = data_rdms_tri,
        mask = mask,         
        results_directory = RESULTS_DIRECTORY
    )

