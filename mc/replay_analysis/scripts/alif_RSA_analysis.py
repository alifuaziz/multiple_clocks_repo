"""
Model RDM script
"""
# Python libraries
# Standard Libraries
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
from pathlib import Path
# RSA specific libraries

# Multiple Clocks Repositiory
import mc.analyse.analyse_MRI_behav     as analyse_MRI_behav
import mc.analyse.extract_and_clean     as extract_and_clean
import mc.simulation.predictions        as predictions
import mc.simulation.RDMs               as RDMs
import mc.replay_analysis.functions.utils      as utils
import mc.replay_analysis.functions.visualise  as v
import mc.replay_analysis.functions.model_rdms as model_rdms
import mc.replay_analysis.functions.data_rdms  as data_rdms


def model_RDM_script(
        **model_rdm_analysis_settings
        ):
    """
    This run on one subject

    Running the whole script to create the model RDM for the subject that is being run

    
    TODO: Look into the corrected RDM function. I Dont think its working properly
    TODO: The RDM is inccorrectly defined.k needs to be fixed

    """
    # Unpack the analysis settings
    sub                     = model_rdm_analysis_settings.get("SUBJECT_NO", "sub-02")
    REGRESSION_VERSION      = model_rdm_analysis_settings.get("REGRESSION_VERSION", "01")
    RDM_VERSION             = model_rdm_analysis_settings.get("RDM_VERSION", "01")

    DATA_DIR                = model_rdm_analysis_settings.get("DATA_DIR", Path("/Users/student/PycharmProjects/data"))
    DERIVATIVES_DIR         = DATA_DIR / "derivatives"

    # 
    TEMPORAL_RESOLUTION     = model_rdm_analysis_settings.get("TEMPORAL_RESOLUTION", 10)
    # The type of model that is being tested in the  RSA analysis (against all of the RSA searchlights) 
    MODEL                   = model_rdm_analysis_settings.get("MODEL", "replay")
    RDM_SIMILARITY_MEASURE  = model_rdm_analysis_settings.get("RDM_SIMILARITY_MEASURE", "pearson")

    # Visuisation Settings
    RDM_VISUALISE           = model_rdm_analysis_settings.get("RDM_VISUALISE", False)
    FMRI_PLOTTING           = model_rdm_analysis_settings.get("FMRI_PLOTTING", False)
    FMRI_SAVE               = model_rdm_analysis_settings.get("FMRI_SAVE", False)


    ####################################################################################################
    # Get the correct working directory

    # Behaviour Direcory
    BEH_dir = DATA_DIR / 'raw' / f'{sub}' / 'beh'
    # RDM Directory
    RDM_dir = DATA_DIR / 'derivatives' / f'{sub}' / 'beh' / f'RDMs_{RDM_VERSION}_glmbase_{REGRESSION_VERSION}'
    # Create the RDM directory
    if not os.path.exists(RDM_dir):
        os.makedirs(RDM_dir)


    # List of Models to analyse
    models_i_want: list = analyse_MRI_behav.models_I_want(RDM_VERSION)

    ####################################################################################################
    # Instantiate dictionaries
    configs_dict = {}
    models_between_task_halves = {}
    
    
    task_halves = ["1", "2"]
    # for each half of the task do the following
    for task_half in task_halves:
        # 1. Extract the behavioural data from the .csv to appropriate dictionaries
            
        # file = data_dir_behav + f"{sub}_fmri_pt{task_half}.csv"
        file = f"{DATA_DIR}/raw/{sub}/beh" + f"/{sub}_fmri_pt{task_half}.csv"

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


    ####################################################################################################
    # Create the model RDMs

    sorted_keys_dict = extract_and_clean.order_task_according_to_rewards(configs_dict)


    # Create the model RDMs
    RDM_object = model_rdms.task_similarity_matrix(
        configs_dict = configs_dict,
        model = MODEL, 
        RDM_dir = RDM_dir,
        VISUALISE = RDM_VISUALISE,
    )


    # Correct for the autocorrelation between task halves. 
    # corrected_RSM_dict = analyse_MRI_behav.auto_corr_RSM_dict(RSM_dict_betw_TH)

    ####################################################################################################

    # # Plot the RDMs
    # if FMRI_PLOTTING == True:
    #     # create directory for saving the RDM images
    #     if not os.path.exists(RDM_dir):
    #         os.makedirs(RDM_dir)

    #     # plot the RDMs from the RSM dictionary
    #     RDMs.plot_RDMs(
    #         RDM_dict = corrected_RSM_dict, 
    #         save_dir = RDM_dir, 
    #         string_for_ticks = sorted_keys_dict['1'])

    # # Save the RDMs
    # if FMRI_SAVE == True: 
    #     # then save these matrices.
    #     if not os.path.exists(RDM_dir):
    #         os.makedirs(RDM_dir)
    #     for RDM in corrected_RSM_dict:
    #         np.save(os.path.join(RDM_dir, f"RSM_{RDM}_{sub}_fmri_both_halves"), corrected_RSM_dict[RDM])

    #     # also save the regression files
    #     for model in models_sorted_into_splits['1']:
    #         np.save(os.path.join(RDM_dir, f"data{model}_{sub}_fmri_both_halves"), np.concatenate(
    #             (models_sorted_into_splits['1'][model], 
    #              models_sorted_into_splits['2'][model]),
    #                 1))
        
    #     # and lastly, save the order in which I put the RDMs.

    #     # save the sorted keys and the regressors.
    #     with open(f"{RDM_dir}/sorted_keys-model_RDMs.pkl", 'wb') as file:
    #         pickle.dump(sorted_keys_dict, file)
        
    #     with open(f"{RDM_dir}/sorted_regs.pkl", 'wb') as file:
    #         pickle.dump(reg_list, file)
            




    pass


if __name__ == "__main__":
    model_RDM_script()