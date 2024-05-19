#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:34:17 2023

from the behavioural files I collected in the experiment, extract behaviour
and use behavioural data to model simulations for the model RDMs. Finally, bin the
simulations according to the GLM I am using for the fMRI data.

28.03.: I am changing something in the preprocessing. This is THE day to change the naming such that it all works well :)

RDM settings (creating the representations):
    01 -> instruction periods, similarity by order of execution, order of seeing, all backw presentations
    01-1 -> instruction periods, location similarity

    01-2 -> Alif: Executution similarity between task halves


    02 -> modelling paths + rewards, creating all possible models
    02-A -> modelling everything but excluding state A
    
    03 -> modelling only reward anchors/rings + splitting clocks model in the same py function.
    03-A -> same as 03 but only considering B,C,D [excluding rew A]

    03-1 -> modelling only reward rings + split ‘clocks model’ = just rotating the reward location around.  
    03-2 -> same as 03-1 but only considering task D and B (where 2 rew locs are the same)
    03-5 - STATE model. only include those tasks that are completely different from all others; i.e. no reversed, no backw. 
    03-5-A -> STATE model. only include those tasks that are completely different from all others; i.e. no reversed, no backw. ; EXCLUDING reward A
    03-99 ->  using 03-1 - reward locations and future rew model; but EVs are scrambled.
    03-999 ->  is debugging 2.0: using 03-1 - reward locations and future rew model; but the voxels are scrambled.
    
    04 -> modelling only paths
    04-5 -> STATE model. only include those tasks that are completely different from all others; i.e. no reversed, no backw.
    04-5-A -> STATE model. only include those tasks that are completely different from all others; i.e. no reversed, no backw. ; EXCLUDING state A


GLM ('regression') settings (creating the 'bins'):
    01 - instruction EVs
    02 - 80 regressors; every task is divided into 4 rewards + 4 paths
    03 - 40 regressors; for every tasks, only the rewards are modelled [using a stick function]
    03-e 40 regressors; for evert task, only take the first 2 repeats.
    03-l 40 regressors; for every task, only take the last 3 repeats.
        careful! sometimes, some trials are not finished and thus don't have any last runs. these are then empty regressors.
    03-2 - 40 regressors; for every task, only the rewards are modelled (in their original time)
    03-3 - 30 regressors; for every task, only the rewards are modelled (in their original time), except for A (because of visual feedback)
    03-4 - 24 regressors; for the tasks where every reward is at a different location (A,C,E), only the rewards are modelled (stick function)
    03-99 - 40 regressors; no button press; I allocate the reward onsets randomly to different state/task combos  -> shuffled through whole task; [using a stick function]
    03-999 - 40 regressors; no button press; created a random but sorted sample of onsets that I am using -> still somewhat sorted by time, still [using a stick function]
    03-9999 - 40 regressors; no button press; shift all regressors 6 seconds earlier
    04 - 40 regressors; for every task, only the paths are modelled
    04-4 - 24 regressors; for the tasks where every reward is at a different location (A,C,E)
    05 - locations + button presses 
    

@author: Svenja Küchenhoff, 2024
"""
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
from pathlib import Path
# RSA specific libraries
# Multiple Clocks Repositiory
import mc
import mc.analyse.analyse_MRI_behav     as analyse_MRI_behav
import mc.analyse.extract_and_clean     as extract_and_clean
import mc.simulation.predictions        as predictions
import mc.simulation.RDMs               as RDMs
import mc.replay_analysis.functions.utils      as utils
import mc.replay_analysis.functions.visualise  as visualise
import mc.replay_analysis.functions.model_rdms as model_rdms
import mc.replay_analysis.functions.data_rdms  as data_rdms


# import pdb; pdb.set_trace()

REGRESSION_VERSION = '01' 
RDM_VERSION        = '01-2' 

if len (sys.argv) > 1:
    SUBJECT_NO = sys.argv[1]
else:
    SUBJECT_NO = '01'

# Behavioural data directory
DATA_DIR = Path("/Users/student/PycharmProjects/data")
DATA_DIR_BEHAV = DATA_DIR / 'raw' / 'sub-{SUBJECT_NO}' / 'beh'

TEMPORAL_RESOLUTION = 10
subjects_list: list = [f"sub-{SUBJECT_NO}"]
task_halves: list = ['1', '2']

# Flag to plot the RDMs
FMRI_PLOTTING = True
# Flag to save the RDMs and the regressors
FMRI_SAVE = True
RDM_SIMILARITY_MEASURE = "pearson" # default is 'pearson'
ADD_RUN_COUNTS_MODEL = False # this doesn't work with the current analysis

# Get the list of the models to analyse  
models_i_want: list = analyse_MRI_behav.models_I_want(RDM_VERSION)
        
for sub in subjects_list:
    # initialize some dictionaries
    models_between_task_halves = {}
    sorted_models_split = {}
    configs_dict = {}
    reg_list = []
    
    # for each half of the task do the following
    for task_half in task_halves:
        # 1. Extract the behavioural data from the .csv to appropriate dictionaries
        # Select the correct file path as a directory for the behavioural .csv file
        DATA_DIR_BEHAV = DATA_DIR / 'raw' / f'{sub}' / 'beh'

        RDM_dir = DATA_DIR / 'derivatives' / f'{sub}' / 'beh' / f'RDMs_{RDM_VERSION}_glmbase_{REGRESSION_VERSION}'

        if os.path.isdir(DATA_DIR_BEHAV):
            print("Running on laptop.")
        else:
            DATA_DIR_BEHAV = DATA_DIR / f'{sub}' / 'beh'
            RDM_dir = '{DATA_DIR}' /'derivatives' / f'{sub}' / 'beh' / 'RDMs_{RDM_VERSION}_glmbase_{REGRESSION_VERSION}'
            print(f"Running on Cluster, setting {DATA_DIR_BEHAV} as data directory")
            
        # file = data_dir_behav + f"{sub}_fmri_pt{task_half}.csv"
        file = f"{DATA_DIR}/raw/sub-01/beh" + f"/{sub}_fmri_pt{task_half}.csv"

        # 1. Extract the behavioural data from the .csv to appropriate dictionaries
        configs, rew_list, rew_index, walked_path, steps_subpath_alltasks_empty, subpath_after_steps, timings, regressors = analyse_MRI_behav.extract_behaviour(file)

        # 2. Create the regressors for the GLM
        # so now, account for the temporal resolution that you want
        for reg in regressors:
            # For each regressor, 
            regressors[reg] = np.repeat(regressors[reg], repeats = TEMPORAL_RESOLUTION)
        
        # Overview of the reward fields per task.
        steps_subpath_alltasks = analyse_MRI_behav.subpath_files(configs, subpath_after_steps, rew_list, rew_index, steps_subpath_alltasks_empty)

        # prepare the between-tasks dictionary.
        all_models_dict = {f"{model}": {key: "" for key in configs} for model in models_i_want}

        # then, lastly, save the all_models_dict in the respective task_half.
        models_between_task_halves[task_half] = all_models_dict
        print(f"task half {task_half}")
        configs_dict[task_half] = rew_list
  
    # out of the between-halves loop
    if RDM_VERSION == '01': # I have to work on this one further for the replay analysis (temporal + spatial)
        # NOTE: this has been fixed in a "hacky" way. by defining the correct dictionary for the RDM_verision = "01". This potentially fine because this function is called when RDM_version == "01"
        models_between_tasks = analyse_MRI_behav.similarity_of_tasks(
            configs_dict, 
            RDM_VERSION = RDM_VERSION) 
        
        models_sorted_into_splits = models_between_tasks.copy()
        
        sorted_keys_dict = extract_and_clean.order_task_according_to_rewards(configs_dict)

    # if the RDM_version is 
    elif RDM_VERSION == '01-2':
        # create dictionary (for both halves, splits) that contains the neural model for each condition being tested
        models_sorted_into_splits = functions_model_rdms.create_Smodel_RDMs(
            configs_dict = configs_dict,
            USE_NEURON_MODEL = True
        )
        # Create sorted_keys_dict for the RDMs
        sorted_keys_dict = extract_and_clean.order_task_according_to_rewards(configs_dict)
    

    # then, in a last step, create the RDMs
    # concatenate the conditions from the two task halves (giving you 2*nCond X nVoxels matrix), 
    # and calculate the correlations between all rows of this matrix. This gives you a symmetric matrix 
    # (of size 2*nCond X 2*nCond), where the (non-symmetric) nCond X nCond bottom left square (or top right, 
    # doesn't matter because it's symmetric) (i.e. a quarter of the original matrix) has all the correlations 
    # across THs. 
    
    # NOTE: the formatting for these dictationaries are the opposite nesting of what is expected from the script.
    # reverse the nesting of the dictionaries.

    # flip the keys and values of the dictionary.
    models_sorted_into_splits = utils.reverse_nested_dict(models_sorted_into_splits)

    # FROM THE NEURAL MODELS, CREATE THE RDMs

    # for each split (half)
    for split in models_sorted_into_splits:
        RSM_dict_betw_TH = {}
        # for each model in the split (half), create the RSM and place it in a dictionary
        for model in models_sorted_into_splits[split]:
            RSM_dict_betw_TH[model] = RDMs.construct_within_task_RSM(
                # The matrix being used is the concatenation of the models between both task halves
                neuron_activation_matrix = np.concatenate(
                    # First half of task (as a vector for each EV)
                    (models_sorted_into_splits['1'][model],
                    # Second half of task
                     models_sorted_into_splits['2'][model]),
                    # the axis along which the concatenation is done
                    1), 
                SIMILARITY_MEASURE = RDM_SIMILARITY_MEASURE,
                plotting    = True, 
                titlestring = model,
                neural_model = False)

            
    # correct the RSM (IDK WHY THOUGH)
    corrected_RSM_dict = analyse_MRI_behav.auto_corr_RSM_dict(RSM_dict_betw_TH)


    # just for me. what happens if I add the ['reward_location', 'one_future_rew_loc' ,'two_future_rew_loc', 'three_future_rew_loc']?
    # addition_model = corrected_RSM_dict['reward_location'] + corrected_RSM_dict['one_future_rew_loc'] + corrected_RSM_dict['two_future_rew_loc'] + corrected_RSM_dict['three_future_rew_loc'] 
    
    # Plot the RDMs
    if FMRI_PLOTTING == True:
        # create directory for saving the RDM images
        if not os.path.exists(RDM_dir):
            os.makedirs(RDM_dir)

        # plot the RDMs from the RSM dictionary
        RDMs.plot_RDMs(
            RDM_dict = corrected_RSM_dict, 
            save_dir = RDM_dir, 
            string_for_ticks = sorted_keys_dict['1'])

    # Save the RDMs
    if FMRI_SAVE == True: 
        # then save these matrices.
        if not os.path.exists(RDM_dir):
            os.makedirs(RDM_dir)
        for RDM in corrected_RSM_dict:
            np.save(os.path.join(RDM_dir, f"RSM_{RDM}_{sub}_fmri_both_halves"), corrected_RSM_dict[RDM])

        # also save the regression files
        for model in models_sorted_into_splits['1']:
            np.save(os.path.join(RDM_dir, f"data{model}_{sub}_fmri_both_halves"), np.concatenate(
                (models_sorted_into_splits['1'][model], 
                 models_sorted_into_splits['2'][model]),
                 1))
        
        # and lastly, save the order in which I put the RDMs.
    
        # save the sorted keys and the regressors.
        with open(f"{RDM_dir}/sorted_keys-model_RDMs.pkl", 'wb') as file:
            pickle.dump(sorted_keys_dict, file)
        
        with open(f"{RDM_dir}/sorted_regs.pkl", 'wb') as file:
            pickle.dump(reg_list, file)
                


