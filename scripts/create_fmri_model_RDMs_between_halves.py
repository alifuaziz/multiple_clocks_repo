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

# RSA specific libraries
# Multiple Clocks Repositiory
import mc
import mc.replay_analysis.utils      as utils
import mc.analyse.analyse_MRI_behav  as analyse_MRI_behav
import mc.analyse.extract_and_clean  as extract_and_clean
import mc.simulation.predictions     as predictions
import mc.simulation.RDMs            as RDMs
import mc.replay_analysis.rdm_models as rdm_models


# import pdb; pdb.set_trace()

REGRESSION_VERSION = '01' 
RDM_VERSION        = '01' 

if len (sys.argv) > 1:
    SUBJECT_NO = sys.argv[1]
else:
    SUBJECT_NO = '01'

# directories
DATA_DIR = "/Users/student/PycharmProjects/data"
DATA_DIR_BEHAV = f"{DATA_DIR}/raw/sub-{SUBJECT_NO}/beh"
# 
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


# import pdb; pdb.set_trace()
        
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
        data_dir_beh = f"{DATA_DIR}/{sub}/beh/"
        RDM_dir = f"{DATA_DIR}/derivatives/{sub}/beh/RDMs_{RDM_VERSION}_glmbase_{REGRESSION_VERSION}"
        if os.path.isdir(data_dir_beh):
            print("Running on laptop.")
        else:
            data_dir_beh = f"{DATA_DIR}/{sub}/beh/"
            RDM_dir = f"{DATA_DIR}/derivatives/{sub}/beh/RDMs_{RDM_VERSION}_glmbase_{REGRESSION_VERSION}"
            print(f"Running on Cluster, setting {data_dir_beh} as data directory")
            
        # file = data_dir_behav + f"{sub}_fmri_pt{task_half}.csv"
        file = f"{DATA_DIR}/raw/sub-01/beh" + f"/{sub}_fmri_pt{task_half}.csv"

        # STEP ONE TAKING PLACE: get the behavioural data I need from the subject files.
        configs, rew_list, rew_index, walked_path, steps_subpath_alltasks_empty, subpath_after_steps, timings, regressors = analyse_MRI_behav.extract_behaviour(file)

        # 2. Create the regressors for the GLM
        # so now, account for the temporal resolution that you want
        for reg in regressors:
            # For each regressor, 
            regressors[reg] = np.repeat(regressors[reg], repeats = TEMPORAL_RESOLUTION)
        
        # overview of the reward fields per task.
        steps_subpath_alltasks = analyse_MRI_behav.subpath_files(configs, subpath_after_steps, rew_list, rew_index, steps_subpath_alltasks_empty)


        # if REGRESSION_VERSION in ['03-4', '03-4-e', '03-4-l','04-4']:
        #     for config in configs:
        #         # remove the tasks that are not needed for the 24 regressors.
        #         if config.startswith('B') or config.startswith('D'):
        #             del rew_list[config]
                
        #     configs = np.array([config for config in configs if config.startswith('A') or config.startswith('C') or config.startswith('E')])

    
        # finally, create simulations and time-bin per run.
        # prepare the between-tasks dictionary.
        all_models_dict = {f"{model}": {key: "" for key in configs} for model in models_i_want}

        # This looks at participants behaviour. depending on the path walked, it creates models based on the similarities of those paths.
        # The behaviour is stored in two different task halve .csv files.
        # if not RDM_VERSION == '01':
        #     for config in configs:
        #         print(f"the config is {rew_list[config]} for {config}")
        #         # select complete trajectory of current task.
        #         trajectory = walked_path[config]
        #         trajectory = [[int(value) for value in sub_list] for sub_list in trajectory]
        #         # select the timings of this task
        #         timings_curr_run = timings[config]
  
        #         # select file that shows step no per subpath
        #         step_number = [[int(value) for value in sub_list] for sub_list in steps_subpath_alltasks[config]]
        #         index_run_no = np.array(range(len(step_number)))
        #         # but only consider some of the repeats for the only later or only early trials!
        #         if REGRESSION_VERSION in ['03-e', '03-4-e']:
        #             # step_number = step_number[0:2].copy()
        #             index_run_no = index_run_no[0:2].copy()
        #         elif REGRESSION_VERSION in ['03-l', '03-4-l']:
        #             # step_number = step_number[2:].copy()
        #             index_run_no = index_run_no[2:].copy()   
                    
        #         # make file that shows cumulative steps per subpath
        #         cumsteps_task = np.cumsum([np.cumsum(task)[-1] for task in step_number])
        
        #         # then start looping through each subpath within one task
        #         repeats_model_dict = {}              
        #         for no_run in index_run_no:
        #             # first check if the run is not completed. if so, skip the uncomplete part.
        #             if len(subpath_after_steps[config]) < 20: # 5 runs a 4 subpaths
        #                 stop_after_x_runs = len(subpath_after_steps[config]) // 4 # 4 subpaths
        #                 if no_run >= stop_after_x_runs:
        #                     continue
                    
        #             if no_run == 0:
        #                 # careful: fields is always one more than the step number
        #                 curr_trajectory = trajectory[0:cumsteps_task[no_run]+1]
        #                 curr_timings = timings_curr_run[0:cumsteps_task[no_run]+1]
        #                 curr_stepnumber = step_number[no_run]
        #             elif no_run > 0:
        #                 # careful: fields is always one more than the step number
        #                 curr_trajectory = trajectory[cumsteps_task[no_run-1]:cumsteps_task[no_run]+1]
        #                 curr_timings = timings_curr_run[cumsteps_task[no_run-1]:cumsteps_task[no_run]+1]
        #                 curr_stepnumber = step_number[no_run]
        #                 curr_cumsumsteps = cumsteps_task[no_run]
                    
        #             # KEY STEP
        #             # Create the moel and place the results in the "repeats_model_dict"
        #             if RDM_VERSION == '01-1': # creating location instruction stuff
        #                 result_model_dict = predictions.create_instruction_model(
        #                     reward_of_task = rew_list[config], 
        #                     trial_type=config
        #                     )
        #             elif RDM_VERSION == '01-2': # creating the model for the execution similarity between task halves
        #                 # result_model_dict = replay_rdm.
        #                 result_model_dict = predictions.create_instruction_model(
        #                     rewards_of_task = rew_list[config],
        #                     trial_type = config
        #                 )
        #             elif RDM_VERSION in ['02', '02-A']: # default, modelling all and splitting clocks.
        #                 result_model_dict = predictions.create_model_RDMs_fmri(
        #                     curr_trajectory, 
        #                     curr_timings, 
        #                     curr_stepnumber, 
        #                     temporal_resolution = TEMPORAL_RESOLUTION,
        #                     plot=False, 
        #                     only_rew = False, 
        #                     only_path= False, 
        #                     split_clock = True)
        #             elif RDM_VERSION in ['03', '03-5', '03-5-A', '03-A']: # modelling only rewards + splitting clocks [new]
        #                 result_model_dict = predictions.create_model_RDMs_fmri(
        #                     curr_trajectory, 
        #                     curr_timings, 
        #                     curr_stepnumber, 
        #                     temporal_resolution = TEMPORAL_RESOLUTION,
        #                     plot=False, 
        #                     only_rew = True, 
        #                     only_path = False, 
        #                     split_clock=True)
        #             elif RDM_VERSION in ['03-1', '03-2', '03-3']:# modelling only clocks + splitting clocks later in different way.
        #                 result_model_dict = predictions.create_model_RDMs_fmri(
        #                     curr_trajectory, 
        #                     curr_timings, 
        #                     curr_stepnumber, 
        #                     temporal_resolution = TEMPORAL_RESOLUTION,
        #                     plot=False, 
        #                     only_rew = True, 
        #                     only_path= False, 
        #                     split_clock = False)    
        #             elif RDM_VERSION in ['04', '04-5-A', '04-A']: # modelling only paths + splitting clocks [new]
        #                 result_model_dict = predictions.create_model_RDMs_fmri(
        #                     curr_trajectory, 
        #                     curr_timings, 
        #                     curr_stepnumber, 
        #                     temporal_resolution = TEMPORAL_RESOLUTION,
        #                     plot=False, 
        #                     only_rew = False, 
        #                     only_path = True, 
        #                     split_clock=True)
                    
                    
        #             # now for all models that are creating or not creating the splits models with my default function, this checking should work.
        #             if RDM_VERSION not in ['03-1', '03-2', '03-3', '03-5', '03-5-A','04-5', '04-5-A']:
        #                 # test if this function gives the same as the models you want, otherwise break!
        #                 model_list = list(result_model_dict.keys())
        #                 if model_list != models_i_want:
        #                     print('careful! the model dictionary did not output your defined models!')
        #                     print(f"These are the models you wanted: {models_i_want}. And these are the ones you got: {model_list}")
        #                     # import pdb; pdb.set_trace() 
                    
        #             # models  need to be concatenated for each run and task
        #             if no_run == 0 or (REGRESSION_VERSION in ['03-l', '03-4-l'] and no_run == 2):
        #                 for model in result_model_dict:
        #                     repeats_model_dict[model] = result_model_dict[model].copy()
        #             else:
        #                 for model in result_model_dict:
        #                     repeats_model_dict[model] = np.concatenate((repeats_model_dict[model], result_model_dict[model]), 1)
                

        #         # NEXT STEP: prepare the regression- select the correct regressors, filter keys starting with 'A1_backw'
        #         regressors_curr_task = {key: value for key, value in regressors.items() if key.startswith(config)}
                
        #         if REGRESSION_VERSION in ['03-e', '03-4-e']:
        #             regressors_curr_task = {regressor: regressors_curr_task[regressor][0:len(repeats_model_dict[model][2])] for regressor in regressors_curr_task}

        #         if REGRESSION_VERSION in ['03-l', '03-4-l']:
        #             regressors_curr_task = {regressor: regressors_curr_task[regressor][(-1*len(repeats_model_dict[model][2])):] for regressor in regressors_curr_task}

        #         print(f"now looking at regressor for task {config}")
                
        #         # check that all regressors have the same length in case the task wasn't completed.
        #         if len(subpath_after_steps[config]) < 20:
        #             # if I cut the task short, then also cut the regressors short.
        #             for reg_type, regressor_list in regressors_curr_task.items():
        #             # Truncate the list if its length is greater than the maximum length
        #                 regressors_curr_task[reg_type] = regressor_list[:(np.shape(repeats_model_dict[list(repeats_model_dict)[0]])[1])]
                
        #         # Ensure all lists have the same length
        #         list_lengths = set(len(value) for value in regressors_curr_task.values())
        #         if len(list_lengths) != 1:
        #             raise ValueError("All lists must have the same length.")
                
        #         # if not all regressors shall be included, filter them according to the regression setting
        #         if REGRESSION_VERSION in ['02']:
        #             if RDM_VERSION == '02-A':
        #                 regressors_curr_task = {key: value for key, value in regressors_curr_task.items() if '_A_' not in key}
        #             else:
        #                 regressors_curr_task = {key: value for key, value in regressors_curr_task.items()}
                
        #         if REGRESSION_VERSION in ['03','03-1','03-2', '03-4', '03-99', '03-999', '03-l', '03-e', '03-4-e', '03-4-l']:
        #             if RDM_VERSION in ['02-A', '03-A', '03-5-A']: # additionally get rid of the A-state.
        #                 regressors_curr_task = {key: value for key, value in regressors_curr_task.items() if '_A_' not in key and key.endswith('reward')}
        #             else:
        #                 regressors_curr_task = {key: value for key, value in regressors_curr_task.items() if key.endswith('reward')}
                
        #         if REGRESSION_VERSION in ['04', '04-4']:
        #             if RDM_VERSION in ['04-5-A', '02-A', '04-A']:
        #                 regressors_curr_task = {key: value for key, value in regressors_curr_task.items() if '_A_' not in key and key.endswith('path')}    
        #             else:
        #                 regressors_curr_task = {key: value for key, value in regressors_curr_task.items() if key.endswith('path')}
     
                    
        #         # sort alphabetically.
        #         sorted_regnames_curr_task = sorted(regressors_curr_task.keys())
        #         # Create a list of lists sorted by keys
        #         sorted_regs = [regressors_curr_task[key] for key in sorted_regnames_curr_task]
        #         regressors_matrix = np.array(sorted_regs)
        #         reg_list.append(sorted_regnames_curr_task)
                
        #         # then do the ORDERED time-binning for each model - across the 5 repeats.
        #         for model in all_models_dict:
        #             if RDM_VERSION == '01-1':
        #                 all_models_dict[model][config] = result_model_dict[model]
        #             # run the regression on all simulated data, except for those as I have a different way of creating them:
        #             elif model not in ['one_future_rew_loc' ,'two_future_rew_loc', 'three_future_rew_loc', 'curr-and-future-rew-locs']:
        #                 all_models_dict[model][config] = mc.simulation.predictions.transform_data_to_betas(repeats_model_dict[model], regressors_matrix)
    
    
        #         # once the regression took place, the location model is the same as the midnight model.
        #         # thus, it will also be the same as predicting future rewards, if we rotate it accordingly!
        #         # temporally not do this
                
        #         if RDM_VERSION in ['03-1', '03-2', '03-3']:
        #             # now do the rotating thing. 
        #             all_models_dict['one_future_rew_loc'][config] = np.roll(all_models_dict['location'][config], -1, axis = 1) 
        #             all_models_dict['two_future_rew_loc'][config] = np.roll(all_models_dict['location'][config], -2, axis = 1) 
        #             if RDM_VERSION in ['03-1', '03-2']:
        #                 all_models_dict['three_future_rew_loc'][config] = np.roll(all_models_dict['location'][config], -3, axis = 1) 
                    
        #             # try something.
        #             all_models_dict['curr-and-future-rew-locs'][config] = np.concatenate(
        #                     (
        #                 all_models_dict['one_future_rew_loc'][config],
        #                 all_models_dict['two_future_rew_loc'][config], 
        #                 all_models_dict['three_future_rew_loc'][config]
        #                     ),  
        #                 0)
                    
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
        models_sorted_into_splits = rdm_models.create_execution_neuron_model(
            configs_dict = configs_dict, 
            RDM_VERSION = RDM_VERSION
        )
        # Create sorted_keys_dict for the RDMs
        sorted_keys_dict = extract_and_clean.order_task_according_to_rewards(configs_dict)
        
    # elif not RDM_VERSION == '01':
    #     # first, sort the models into two equivalent halves, just in case this went wrong before.
    #     # DOUBLE CHECK IF THIS SORTING ACTUALLY WORKS!!!!
        
    #     sorted_keys_dict = extract_and_clean.order_task_according_to_rewards(configs_dict)
    #     models_sorted_into_splits = {task_half: {model: {config: "" for config in sorted_keys_dict[task_half]} for model in models_i_want} for task_half in task_halves}
    #     test = {task_half: {model: "" for model in models_i_want} for task_half in task_halves}
        
    #     for half in models_between_task_halves:
    #         for model in models_between_task_halves[half]:
    #             for task in models_between_task_halves[half][model]:
    #                 if task in sorted_keys_dict['1']:
    #                     models_sorted_into_splits['1'][model][task] = models_between_task_halves[half][model][task]
    #                 elif task in sorted_keys_dict['2']:
    #                     models_sorted_into_splits['2'][model][task] = models_between_task_halves[half][model][task]                
    #     # then, do the concatenation across the ordered tasks.
    #     # import pdb; pdb.set_trace()
    #     for split in models_sorted_into_splits:
    #         for model in models_sorted_into_splits[split]:
    #             test[split][model] = np.concatenate([models_sorted_into_splits[split][model][task] for task in sorted_keys_dict[split]], 1)

    #             models_sorted_into_splits[split][model] = np.concatenate([models_sorted_into_splits[split][model][task] for task in sorted_keys_dict[split]], 1)  

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
            # mc.simulation.predictions.plot_without_legends(RSM_dict_betw_TH[model])
            # if RDM_VERSION in ['03-5', '03-5-A', '04-5', '04-5-A']:
            #     if RDM_VERSION in ['03-5','04-5']:
            #         exclude = 0
            #     elif RDM_VERSION in ['03-5-A', '04-5-A']:
            #         exclude = 1
            #     # this is really just playing around. the proper work will be in fmri_do_RSA!!!
            #     RSM_dict_betw_TH_mask = predictions.create_mask_same_tasks(
            #         RSM_dict_betw_TH[model], 
            #         configs_dict, 
            #         exclude
            #         )
            #     # import pdb; pdb.set_trace()
            #     RSM_dict_betw_TH['state_masked'] = np.where(RSM_dict_betw_TH_mask == 1, RSM_dict_betw_TH[model], np.nan)
            #     # import pdb; pdb.set_trace()
            #     #RSM_dict_betw_TH['state_masked'] = RSM_dict_betw_TH[model]* RSM_dict_betw_TH_mask
            
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
        if RDM_VERSION in ['03-5', '03-5-A', '04-5', '04-5-A']:
            np.save(os.path.join(RDM_dir, f"RSM_state_mask_across_halves"), RSM_dict_betw_TH_mask)
        for RDM in corrected_RSM_dict:
            # Saves RDM (array) as a .npy file
            np.save(os.path.join(RDM_dir, f"RSM_{RDM}_{sub}_fmri_both_halves"), corrected_RSM_dict[RDM])
            
        # also save the regression files
        for model in models_sorted_into_splits['1']:
            np.save(os.path.join(RDM_dir, f"data{model}_{sub}_fmri_both_halves"), np.concatenate(
                (models_sorted_into_splits['1'][model], 
                 models_sorted_into_splits['2'][model]),
                 1))
        
        # and lastly, save the order in which I put the RDMs.
    
        # if the variable exists
        # save the sorted keys and the regressors.

        with open(f"{RDM_dir}/sorted_keys-model_RDMs.pkl", 'wb') as file:
            pickle.dump(sorted_keys_dict, file)
        
        with open(f"{RDM_dir}/sorted_regs.pkl", 'wb') as file:
            pickle.dump(reg_list, file)
                


