#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:25:52 2023

create fMRI data RDMs

28.03.: I am changing something in the preprocessing. This is THE day to change the naming such that it all works well :)

RDM settings (creating the representations):
    01 -> instruction periods, similarity by order of execution, order of seeing, all backw presentations
    01-1 -> instruction periods, location similarity
    
    01-2 ->

    02 -> modelling paths + rewards, creating all possible models
    
    03 -> modelling only reward anchors/rings + splitting clocks model in the same py function.
    03-1 -> modelling only reward rings + split ‘clocks model’ = just rotating the reward location around. 
    03-2 -> same as 03-1 but only considering task D and B (where 2 rew locs are the same)
    03-3 -> same as 03-1 but only considering B,C,D [excluding rew A] -> important to be paired with GLM 03-3!
    03-5 - STATE model. only include those tasks that are completely different from all others; i.e. no reversed, no backw. 
    03-5-A -> STATE model. only include those tasks that are completely different from all others; i.e. no reversed, no backw. ; EXCLUDING reward A
    03-99 ->  using 03-1 - reward locations and future rew model; but EVs are scrambled.
    03-999 ->  is debugging 2.0: using 03-1 - reward locations and future rew model; but the voxels are scrambled.    
    
    04 -> modelling only paths
    04-5-A -> STATE model. only include those tasks that are completely different from all others; i.e. no reversed, no backw. ; EXCLUDING reward A
    
    xx-999 ->  is debugging 2.0: using whatever, but the voxels are scrambled.


GLM ('regression') settings (creating the 'bins'):
    01 - instruction EVs
    02 - 80 regressors; every task is divided into 4 rewards + 4 paths
    03 - 40 regressors; for every tasks, only the rewards are modelled [using a stick function]
    03-2 - 40 regressors; for every task, only the rewards are modelled (in their original time)
    03-3 - 30 regressors; for every task, only the rewards are modelled (in their original time), except for A (because of visual feedback)
    03-4 - 24 regressors; for the tasks where every reward is at a different location (A,C,E), only the rewards are modelled (stick function)
        Careful! I computed one data (subject-level) GLM called 03-4. This is simply a 03 without button presses!
        Not the same as 03-4 in this sense, but ok to be used.
    03-99 - 40 regressors; no button press; I allocate the reward onsets randomly to different state/task combos  -> shuffled through whole task; [using a stick function]
    03-999 - 40 regressors; no button press; created a random but sorted sample of onsets that I am using -> still somewhat sorted by time, still [using a stick function]
    03-9999 - 40 regressors; no button press; shift all regressors 6 seconds earlier
    04 - 40 regressors; for every task, only the paths are modelled
    04-4 - 24 regressors; for the tasks where every reward is at a different location (A,C,E)
    05 - locations + button presses 
    
@author: Svenja Küchenhoff, 2024
"""
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
from rsatoolbox.util.searchlight import get_volume_searchlight, get_searchlight_RDMs

# mc imports
import mc.analyse.analyse_MRI_behav     as analyse_MRI_behav
import mc.analyse.calc                  as calc
import mc.replay_analysis.functions_model_rdms as functions_model_rdms


REGRESSION_VERSION = '01' 
RDM_VERSION        = '01-2' 


if len (sys.argv) > 1:
    SUB_NO = sys.argv[1]
else:
    SUB_NO = '01'



USE_PREVIOUS_SEARCHLIGHTS = True  # Searchlights are loaded from file
USE_PREVIOUS_DATA_RDM     = False # Data RDMs are loaded from file
VISUALISE_RDMS            = False # Visualise the data RDMs
REMOVE_AUTOCORR           = True  # Remove autocorrelations from the data RDMs, else cross-correlate

# The parts of the BOLD signal that areused 
EVS_TYPE = "instruction_period"


data_folder = '/Users/student/PycharmProjects/data'
data_folder = Path(data_folder)


subject_list: list = [f"sub-{SUB_NO}"]
#subjects = ['sub-01']
task_halves: list = ['1', '2']

print(f"Now running RSA for RDM version {RDM_VERSION} based on subj GLM {REGRESSION_VERSION} for subj {SUB_NO}")

# get the list of the models to be analysed
models_I_want = analyse_MRI_behav.models_I_want(RDM_VERSION)

# based on GLM, get the number of conditions in the RDM
# for some reason this function is not working!
no_RDM_conditions: int = analyse_MRI_behav.get_no_RDM_conditions(RDM_VERSION)

    
for sub in subject_list:
    # GETTING THE CORRECT DATA DIRECTORY
    subject_directory = data_folder / 'derivatives' / sub
    if os.path.isdir(subject_directory):
        print("Running on laptop.")
    else:
        subject_directory = f"{data_folder}/derivatives/{sub}"
        print(f"Running on Cluster, setting {subject_directory} as data directory")
    
    # 
    if RDM_VERSION in ['03-999']:
        RDM_dir = f"{subject_directory}/beh/RDMs_03_glmbase_{REGRESSION_VERSION}"
    else:
        RDM_dir = subject_directory / 'beh' / f"RDMs_{RDM_VERSION}_glmbase_{REGRESSION_VERSION}"
    
    
    # Make the RDM_dir if it doesn't exist
    if not RDM_dir.exists():
        RDM_dir.mkdir(parents=True)
    results_dir = subject_directory / 'func' / f"RSA_{RDM_VERSION}_glmbase_{REGRESSION_VERSION}"
    
    # Make the results_dir if it doesn't exist
    if not results_dir.exists():
        results_dir.mkdir(parents=True)
        os.makedirs(f"{results_dir}/results")
    # results_dir = f"{subject_directory}/func/RSA_{RDM_VERSION}_glmbase_{REGRESSION_VERSION}/results" 
    # if os.path.exists(results_dir):
    #     # move pre-existing files into a different folder.
    #     analyse_MRI_behav.move_files_to_subfolder(results_dir)
    # get a reference image to later project the results onto. This is usually
    # example_func from half 1, as this is where the data is corrected to.
    
    results_dir = results_dir / 'results' 
    ref_img = load_img( subject_directory / 'func' / 'preproc_clean_01.feat'/ 'example_func.nii.gz' )
    
    # load the file which defines the order of the model RDMs, and hence the data RDMs
    with open(f"{RDM_dir}/sorted_keys-model_RDMs.pkl", 'rb') as file:
        sorted_keys = pickle.load(file)
    with open(f"{RDM_dir}/sorted_regs.pkl", 'rb') as file:
        reg_keys = pickle.load(file)
    # also store 2 dictionaries of the EVs


    REGRESSION_VERSION = analyse_MRI_behav.preprocess_regression_version(REGRESSION_VERSION)
    
    # create dictionary of paths to the EVs for each half (split) of the task
    # first half
    EV_path_dict_01 = analyse_MRI_behav.get_EV_dict(
        subject_directory, REGRESSION_VERSION
        )
    # second half
    EV_path_dict_02 = analyse_MRI_behav.get_EV_dict(
        subject_directory, REGRESSION_VERSION
        )


    # Step 1: get the searchlights
    mask = load_img(f"{subject_directory}/anat/{sub}_T1w_noCSF_brain_mask_bin_func_01.nii.gz")

    centers, neighbors = functions_model_rdms.get_searchlights(
        mask = mask,
        radius = 3, 
        threshold = 0.5,
        USE_PREVIOUS_SEARCHLIGHTS = USE_PREVIOUS_SEARCHLIGHTS,
        NEIGHBORS_PATH=f"{RDM_dir}/searchlight_neighbors.pkl",
        CENTERS_PATH=f"{RDM_dir}/searchlight_centers.pkl",
        SAVE_SEARCHLIGHTS = True
    )

    # Step 2: loading and computing the data RDMs
    if USE_PREVIOUS_DATA_RDM == True:
        # Open the data RDM file
        with open(f"{results_dir}/data_RDM.pkl", 'rb') as file:
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

        for split in task_halves:

            EVs_path_dict = functions_model_rdms.get_EV_path_dict_instruction_period(
                subject_directory = subject_directory,
                split = split
                )
            
            # Load in the EVs for the instruction periods from the dictionary of paths
            EVs_data_dict = functions_model_rdms.load_EV_data(
                EVs_path_dict = EVs_path_dict,
                RDM_VERSION = RDM_VERSION
            )

            # Convert the dictionary of EVs to a 2D array
            EVs_data_2d = functions_model_rdms.EV_data_dict_to_2d(
                EVs_data_dict = EVs_data_dict,
            )
            
            # Put the 2D array of EVs into the EV_both_halves dictionary
            EVs_both_halves_dict[split] = EVs_data_dict
            EVs_both_halves_2d[split] = EVs_data_2d

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
        data_conds = np.reshape(np.tile((np.array(['cond_%02d' % x for x in np.arange(no_RDM_conditions)])), (1, 2)).transpose(), 2 * no_RDM_conditions)

        # Defining both task halves runs: 0s first half, 1s is second half
        sessions = np.concatenate((np.zeros(int(EVs_both_halves_2d['1'].shape[0])),
                                    np.ones(int(EVs_both_halves_2d['2'].shape[0]))))  
        


        # for all other cases, cross correlated between task-halves.
        # TODO: try to have one only half 
        data_RDM = get_searchlight_RDMs(
            data_2d = EVs_both_halves_2d, 
            centers = centers, 
            neighbors = neighbors, 
            events  = data_conds, 
            method  ='crosscorr', 
            cv_descr = sessions
                        )
        
        # Save the data RDMs
        with open(f"{results_dir}/data_RDM.pkl", 'wb') as file:
            pickle.dump(data_RDM, file)

    # Step 3: Load  Model RDMs, created in `replay_RSA_model_RDM.py`
    neuron_model_RDMs = functions_model_rdms.load_model_RDMs(
        RDM_dir = RDM_dir,
        models_I_want = models_I_want,
        sub = sub,
        )
    
    # Step 4: Create Fixed Model RDMs

    def prepare_model_RDM_dict(
        model_RDMs: dict,
        no_RDM_conditions: int,
        RDM_VERSION: str,
        MODEL_TYPE: str = "matrix",
        VISUALISE_RDMS: bool = False,
        REMOVE_AUTOCORR: bool = True
    ):
        """
        Parameters
            neuron_model_RDMs: dictionary of model RDMs
            no_RDM_conditions: number of conditions in the RDM
            RDM_VERSION: version of the RDM
            MODEL_TYPE: "neuron" represent the vector form of the model RDM. "matrix" represents the matrix form of the model RDM
            REMOVE_AUTOCORR: remove autocorrelations from the model RDMs. If True it takes into account to the two task halves

        Returns
            model_RDM_dir: dict Dictionary of model RDMs
        """

        # Where, each model gets its own, separate estimation.
        model_RDM_dir = {}

        if MODEL_TYPE == "neuron":
            # Use the vectors for the different conditions to create the RDMs

            for model in model_RDMs:
                # 4.1 either prepare the neuron_modlel and place it into the `analyse_MRI_behav.prepare_model_data function`, then run the `rsr.calc_rdm` function
                # prepare the neuron_model of the data into the Dataset object that is used in rsr.calc_rdm()
                model_data = analyse_MRI_behav.prepare_model_data(
                    model_RDMs[model],
                    no_RDM_conditions,
                    RDM_VERSION)
                
                if REMOVE_AUTOCORR == False:
                    # assumes that both task halves are one session
                    model_RDM_dir[model] = rsr.calc_rdm(
                        model_data, 
                        method='correlation', 
                        descriptor='conds'
                        )

                else:
                    # assumes that there are two task halves, that need to be corrected for autocorrelations
                    model_RDM_dir[model] = rsr.calc_rdm(
                        model_data, 
                        method='crosscorr', 
                        descriptor='conds', 
                        cv_descriptor='sessions'
                        )

        elif MODEL_TYPE == "matrix":
            # takes a dictionary of matrix RDMs as the correctly packaged into the RDMs object in a dictionary

            for model in model_RDMs:
                
                # get the model from the model_RDMs dictionary
                model_RDM = model_RDMs[model]

                if AUTO_CORR == False:
                    model_RDM_dir[model] = rsr.calc_rdm(
                        dataset = model_RDM,
                        method = 'correlation',
                        descriptor = 'conds'
                    )
                
                else: 
                    model_RDM_dir[model] = rsr.calc_rdm(
                        dataset = model_RDM,
                        method = 'crosscorr',
                        descriptor = 'conds',
                        cv_descriptor = 'sessions'
                    )

        if VISUALISE_RDMS == True:
            # Visualise the model RDMs
            fig, ax, ret_vla = rsatoolbox.vis.show_rdm(model_RDM_dir[model])


        return model_RDM_dir

    # Prepare the model RDMs
    model_RDM_dir = prepare_model_RDM_dict(
        model_RDMs = neuron_model_RDMs,
        no_RDM_conditions = no_RDM_conditions,
        RDM_VERSION = RDM_VERSION,
        MODEL_TYPE = "neuron",
        VISUALISE_RDMS = VISUALISE_RDMS,
        REMOVE_AUTOCORR = REMOVE_AUTOCORR
    )


    def run_fixed_model(
        model_RDM_dir: dict
    ):
        """

        # Step 4.2: evaluate the model fit between model and data RDMs.
        """
        # Dictionary to store the results
        RDM_my_model_dir = {}

        for model in model_RDM_dir:
            # Define the type of model to be evaluated. It is a single model, not a set of models that. It does not have a set of betas to also fit. 
            single_model = rsatoolbox.model.ModelFixed(f"{model}_only", model_RDM_dir[model])
            # Run the model evaluation for all searchlights
            RDM_my_model_dir[model] = Parallel(n_jobs=3)(delayed(analyse_MRI_behav.evaluate_model)(single_model, data_RDM_p_voxel) for data_RDM_p_voxel in tqdm(data_RDM, desc=f"running GLM for all searchlights in {model}"))
            

        # return dictionary of results
        return RDM_my_model_dir


    # Run the fixed model
    RDM_my_model_dir = run_fixed_model(
        model_RDM_dir = model_RDM_dir
    )

    # Step 4.3: Save the results
    for model in RDM_my_model_dir:
        # Save the different aspects of the model as different nii files 
        analyse_MRI_behav.save_RSA_result(
                            result_file = RDM_my_model_dir[model], 
                            data_RDM_file = data_RDM, 
                            file_path = results_dir, 
                            file_name = f"{model}", 
                            mask = mask, 
                            number_regr = 0, 
                            ref_image_for_affine_path=ref_img
                            )
            


    # Step 5: Create a model that is a combination of models.