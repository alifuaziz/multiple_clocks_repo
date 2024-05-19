"""
Data RDM Script

File that contains one script that runs the data RDM analysis for a single subject.

"""
# Python libraries
# Standard Libraries
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import random
import os
from pathlib import Path
from joblib import Parallel, delayed

# RSA specific libraries
import nibabel as nib
from nilearn.image import load_img
import rsatoolbox.rdm as rsr
import rsatoolbox.vis as vis
import rsatoolbox
from rsatoolbox.rdm import RDMs
from rsatoolbox.util.searchlight import get_volume_searchlight, get_searchlight_RDMs

# mc imports
import mc.analyse.analyse_MRI_behav     as analyse_MRI_behav
import mc.analyse.calc                  as calc
import mc.replay_analysis.functions.model_rdms as model_rdm_functions



def data_RDM_script(
    **data_rdm_analysis_settings
    ):  
    """
    
    """
    # Unpack the analysis settings
    sub                     = data_rdm_analysis_settings.get("SUBJECT_NO", "sub-01")
    REGRESSION_VERSION      = data_rdm_analysis_settings.get("REGRESSION_VERSION", "01")
    RDM_VERSION             = data_rdm_analysis_settings.get("RDM_VERSION", "01")

    DATA_DIR                = data_rdm_analysis_settings.get("DATA_DIR", Path("/Users/student/PycharmProjects/data"))
    SUBJECT_DIRECTORY = DATA_DIR / 'derivatives' / sub
    print(SUBJECT_DIRECTORY)
    if os.path.isdir(SUBJECT_DIRECTORY):
        print("Running on laptop.")
    DERIVATIVES_DIR         = DATA_DIR / "derivatives"
    RDM_DIR = SUBJECT_DIRECTORY / 'beh' / f"RDMs_{RDM_VERSION}_glmbase_{REGRESSION_VERSION}"
    # Make the RDM_dir if it doesn't exist
    if not RDM_DIR.exists():
        RDM_DIR.mkdir(parents=True)
    RESULTS_DIR = SUBJECT_DIRECTORY / 'func' / f"RSA_{RDM_VERSION}_glmbase_{REGRESSION_VERSION}"



    EVS_TYPE = data_rdm_analysis_settings.get("EVS_TYPE", "instruction_period")
    REMOVE_AUTOCORR = data_rdm_analysis_settings.get("REMOVE_AUTOCORR", False)
    TASK_HALVES = data_rdm_analysis_settings.get("TASK_HALVES", ['1', '2'])
    
    USE_PREVIOUS_SEARCHLIGHTS = data_rdm_analysis_settings.get("USE_PREVIOUS_SEARCHLIGHTS", False)
    USE_PREVIOUS_DATA_RDM = data_rdm_analysis_settings.get("USE_PREVIOUS_DATA_RDM", False)

    VISUALISE_RDMS = data_rdm_analysis_settings.get("VISUALISE_RDMS", False)
    
    ####################################################################################################
    # Get the correct working directory

    # results_dir = results_dir / 'results' 
    ref_img = load_img( SUBJECT_DIRECTORY / 'func' / 'preproc_clean_01.feat'/ 'example_func.nii.gz' )

    # load the file which defines the order of the model RDMs, and hence the data RDMs
    with open(f"{RDM_DIR}/sorted_keys-model_RDMs.pkl", 'rb') as file:
        sorted_keys = pickle.load(file)
    with open(f"{RDM_DIR}/sorted_regs.pkl", 'rb') as file:
        reg_keys = pickle.load(file)
    # also store 2 dictionaries of the EVs


    ####################################################################################################


    REGRESSION_VERSION = analyse_MRI_behav.preprocess_regression_version(REGRESSION_VERSION)

    # create dictionary of paths to the EVs for each half (split) of the task
    # first half
    EV_path_dict_01 = analyse_MRI_behav.get_EV_dict(
        SUBJECT_DIRECTORY, REGRESSION_VERSION
        )
    # second half
    EV_path_dict_02 = analyse_MRI_behav.get_EV_dict(
        SUBJECT_DIRECTORY, REGRESSION_VERSION
        )



    ####################################################################################################
    # Get Searchlights
    # Step 1: get the searchlights

    # Load the functional mask of the subject (i.e. the brain mask of only the computation parts of the brain)
    mask = load_img(f"{SUBJECT_DIRECTORY}/anat/{sub}_T1w_noCSF_brain_mask_bin_func_01.nii.gz")

    # Get the searchlights
    centers, neighbors = model_rdm_functions.get_searchlights(
        mask = mask,
        radius = 3, 
        threshold = 0.5,
        USE_PREVIOUS_SEARCHLIGHTS = USE_PREVIOUS_SEARCHLIGHTS,
        NEIGHBORS_PATH=f"{RDM_DIR}/searchlight_neighbors.pkl",
        CENTERS_PATH=f"{RDM_DIR}/searchlight_centers.pkl",
        SAVE_SEARCHLIGHTS = True
    )

    ####################################################################################################
    # Step 2: loading and computing the data RDMs
    if USE_PREVIOUS_DATA_RDM == True:
        # Open the data RDM file
        with open(f"{RESULTS_DIR}/data_RDM.pkl", 'rb') as file:
            data_RDM = pickle.load(file)
        
        if VISUALISE_RDMS == True:
            analyse_MRI_behav.visualise_data_RDM(mni_x = 53, 
                                                mni_y = 30, 
                                                mni_z = 2, 
                                                data_RDM_file = data_RDM, 
                                                mask = mask)
            
    else:
        # Create dictionary to store the data for each EV for both task halves
        EVs_both_halves_dict = {
            '1': None,
            '2': None
        }
        # create new dictionary to store the 2D array of EVs for both task halves
        EVs_both_halves_2d = EVs_both_halves_dict.copy()

        for split in TASK_HALVES:

            
            EVs_path_dict = model_rdm_functions.get_EV_path_dict(
                subject_directory = SUBJECT_DIRECTORY,
                split = split,
                EVs_type = EVS_TYPE
                )
            
            # Load in the EVs for the instruction periods from the dictionary of paths
            EVs_data_dict = model_rdm_functions.load_EV_data(
                EVs_path_dict = EVs_path_dict,
                RDM_VERSION = RDM_VERSION
            )

            # Convert the dictionary of EVs to a 2D np.array (10 * 746496) (10 conditions, 746496 voxels)
            EVs_data_2d = model_rdm_functions.EV_data_dict_to_2d(
                EVs_data_dict = EVs_data_dict,
            )
            
            # Put the 2D array of EVs into the EV_both_halves dictionary
            EVs_both_halves_dict[split] = EVs_data_dict
            EVs_both_halves_2d  [split] = EVs_data_2d

            # Each part has array of shape (n_conditions, x, y, z)
            EVs_both_halves_dict[split] = np.array(list(EVs_both_halves_dict[split].values()))
            # Remove NaNs
            EVs_both_halves_dict[split] = np.nan_to_num(EVs_both_halves_dict[split])

            # Each part has array of shape (n_conditions, n_voxels)
            EVs_both_halves_2d[split] = EVs_both_halves_dict[split].reshape(EVs_both_halves_dict[split].shape[0], -1)

        # Combine the EVs from both task halves into a single 2D array (condition, x, y, z)
        EVs_both_halves_array_2d = np.concatenate((EVs_both_halves_2d['1'], EVs_both_halves_2d['2']), axis=0)

        # Remove NaNs
        # EVs_both_halves_array_2d = np.nan_to_num(EVs_both_halves_array_2d)

        # define the condition names for both task halves
        # 2 * 10 conditions, since there are 10 identical executution conditions in each task half
        # data_conds = np.reshape(np.tile((np.array(['cond_%02d' % x for x in np.arange(no_RDM_conditions)])), (1, 2)).transpose(), 2 * no_RDM_conditions)
        data_conds = [x for x in range(20)]
        # data_conds = np.reshape(np.tile((np.array(['cond_%02d' % x for x in np.arange(no_RDM_conditions)])), (1)).transpose(), no_RDM_conditions)

        # Defining both task halves runs: 0s first half, 1s is second half
        sessions = np.concatenate((np.zeros(int(EVs_both_halves_2d['1'].shape[0])),   # 10 of each condition
                                    np.ones(int(EVs_both_halves_2d['2'].shape[0]))))  # 10 of each condition
        

    ####################################################################################################
    # for all other cases, cross correlated between task-halves.
    # TODO: try to have one only half 
    data_RDM = get_searchlight_RDMs(
        # data_2d = EVs_both_halves_2d,         # (nObs x nVox) (20 * 746496)
        data_2d = EVs_both_halves_array_2d, # (nObs x nVox)
        centers = centers, 
        neighbors = neighbors, 
        events  = data_conds,                 # (nObs x 1) of condition labels (con§d_00, cond_01, ... cond_09)
        method = 'correlation', 
        # method  ='crosscorr', 
        # cv_descr = sessions                   # (nObs x 1) of session labels (0, 1)
                    )

    # Save the data RDMs
    with open(f"{RESULTS_DIR}/data_RDM.pkl", 'wb') as file:
        pickle.dump(data_RDM, file)

    ####################################################################################################
    # Load in the Model RDMs object from Previous Script
    replay_dir = f"/Users/student/PycharmProjects/data/derivatives/{sub}/beh/RDMs_01_glmbase_01/replay_RDM_object.pkl"

    with open(replay_dir, 'rb') as file:
        replay_RDM_object = pickle.load(file)

    ####################################################################################################
    # Not present: Create the Model RDM from the vector models of the response

    ####################################################################################################
    # Dictionary to store the results
    RDM_my_model_dir = {}
    # Define the type of model to be evaluated. It is a single model, not a set of models that. It does not have a set of betas to also fit. 
    model = "replay"
    single_model = rsatoolbox.model.ModelFixed(f"{model}_only", replay_RDM_object)

    # Run the model evaluation for all searchlights
    RDM_my_model_dir[model] = Parallel(n_jobs=3)(delayed(analyse_MRI_behav.evaluate_model)(single_model, data_RDM_p_voxel) for data_RDM_p_voxel in tqdm(data_RDM, desc=f"running GLM for all searchlights in {model}"))

    ####################################################################################################

    # Step 4.3: Save the results
    for model in RDM_my_model_dir:
        # Save the different aspects of the model as different nii files 
        analyse_MRI_behav.save_RSA_result(
                            result_file = RDM_my_model_dir[model], 
                            data_RDM_file = data_RDM, 
                            file_path = RESULTS_DIR, 
                            file_name = f"{model}", 
                            mask = mask, 
                            number_regr = 0, 
                            ref_image_for_affine_path=ref_img
                            )
            



if __name__ == "__main__":
    data_RDM_script()


