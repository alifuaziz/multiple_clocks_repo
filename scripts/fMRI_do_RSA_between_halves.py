#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:25:52 2023

create fMRI data RDMs


@author: xpsy1114
"""


from tqdm import tqdm
import numpy as np
import nibabel as nib
import os
import re
import rsatoolbox.rdm as rsr
import rsatoolbox
from rsatoolbox.util.searchlight import get_volume_searchlight, get_searchlight_RDMs
from nilearn.image import load_img
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import mc
import pickle
import sys

# import pdb; pdb.set_trace()  

if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '01'

subjects = [f"sub-{subj_no}"]
load_old = True


#subjects = ['sub-01']
task_halves = ['1', '2']
RDM_version = '09' # 10 is like 09 but leaving out the A-State.

# 09 is reward location and future reward location.
# 07 is the reward based model
# 06 is both task halves combined and only looking at reward times.
# RDM_version = '05' # 05 is both task halves combined
# 04 is another try to bring the results back...'03' # 03 is teporal resolution = 1. 02 is for the report.

# models_I_want = ['reward_midnight', 'reward_clocks', 'state', 'task_prog']

if RDM_version == '05': 
    models_I_want = ['location', 'phase', 'phase_state', 'midnight', 'clocks', 'state', 'task_prog']
elif RDM_version == '06':
    models_I_want = ['reward_midnight', 'reward_clocks', 'state', 'task_prog']
elif RDM_version == '07':
    models_I_want = ['reward_midnight_v2', 'reward_clocks_v2', 'state', 'task_prog']
elif RDM_version == '08':
    models_I_want = ['reward_midnight_v2', 'reward_clocks_v2', 'state', 'task_prog']
elif RDM_version == '09' or RDM_version == '10':
    models_I_want = ['reward_location', 'one_future_rew_loc' ,'two_future_rew_loc', 'three_future_rew_loc', 'reward_midnight_v2', 'reward_clocks_v2']


regression_version = '08' # 08 is rewards only and without A (because of the visual feedback)
# 07 is only button press and rewards.
# regression_version = '06' new, better script is now 06 #'04_pt01+_that_worked' 
# make all paths relative and adjust to both laptop and server!!
no_RDM_conditions = 80

if regression_version == '07' or regression_version == '06':
    no_RDM_conditions = 40
    
elif regression_version == '08':
    no_RDM_conditions = 30
    
for sub in subjects:
    data_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}"
    if os.path.isdir(data_dir):
        print("Running on laptop.")
    else:
        data_dir = f"/home/fs0/xpsy1114/scratch/data/derivatives/{sub}"
        print(f"Running on Cluster, setting {data_dir} as data directory")
       
    RDM_dir = f"{data_dir}/beh/RDMs_{RDM_version}_glmbase_{regression_version}"
    if not os.path.exists(RDM_dir):
        os.makedirs(RDM_dir)
        
    results_dir = f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version}"   
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        os.makedirs(f"{results_dir}/results")
    results_dir = f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version}/results"  
    
    # get a reference image to later project the results onto. This is usually
    # example_func from half 1, as this is where the data is corrected to.
    ref_img = load_img(f"{data_dir}/func/preproc_clean_01.feat/example_func.nii.gz")
    
    # Step 1: creating the searchlights
    # mask will define the searchlight positions, in pt01 space because that is 
    # where the functional files have been registered to.
    mask = load_img(f"{data_dir}/anat/grey_matter_mask_func_01.nii.gz")
    mask = mask.get_fdata()  
    # save this file to save time
    if load_old:
        with open(f"{RDM_dir}/searchlight_centers.pkl", 'rb') as file:
            centers = pickle.load(file)
        with open(f"{RDM_dir}/searchlight_neighbors.pkl", 'rb') as file:
            neighbors = pickle.load(file)
        #centers = np.load(f"{RDM_dir}/searchlight_centers.npy", allow_pickle=True)
        #neighbors = np.load(f"{RDM_dir}/searchlight_neihbors.npy", allow_pickle=True)
    else:
        # creating the searchlights
        centers, neighbors = get_volume_searchlight(mask, radius=3, threshold=0.5) # Found 175.483 searchlights
        # if I use the grey matter mask, then I find 144.905 searchlights
        # save this structure
        with open(f"{RDM_dir}/searchlight_centers.pkl", 'wb') as file:
            pickle.dump(centers, file)
        with open(f"{RDM_dir}/searchlight_neighbors.pkl", 'wb') as file:
            pickle.dump(neighbors, file)   
        #np.save(f"{RDM_dir}/searchlight_centers.npy", centers)
        #np.save(f"{RDM_dir}/searchlight_neihbors.npy", neighbors)
            
    # Step 2: loading and computing the data RDMs
    if load_old:
        with open(f"{results_dir}/data_RDM.pkl", 'rb') as file:
            data_RDM = pickle.load(file)
    else:
        data_RDM_file_2d = {}
        data_RDM_file = {}
        for task_half in task_halves:
            # import pdb; pdb.set_trace()  
            # load the relevant pre-processed task-half
            fmri_data_dir = f"{data_dir}/func/preproc_clean_0{task_half}.feat"
            pe_path = f"{data_dir}/func/glm_{regression_version}_pt0{task_half}.feat/stats"
            # define the naming conventions in this folder
            data_RDM_file[task_half] = [None] * no_RDM_conditions  # Initialize a list
            image_paths = [None] * no_RDM_conditions
            for reg_index in range(1, no_RDM_conditions+1):
                file_path = os.path.join(pe_path, f"pe{reg_index}.nii.gz")
                image_paths[reg_index-1] = file_path  # save path to check if everything went fine later
                data_RDM_file[task_half][reg_index-1] = nib.load(file_path).get_fdata()

            # Convert the list to a NumPy array
            data_RDM_file[task_half] = np.array(data_RDM_file[task_half])
            # reshape data so we have n_observations x n_voxels
            data_RDM_file_2d[task_half] = data_RDM_file[task_half].reshape([data_RDM_file[task_half].shape[0], -1])
            data_RDM_file_2d[task_half] = np.nan_to_num(data_RDM_file_2d[task_half]) # now this is 80timepoints x 746.496 voxels
    
        # define the conditions, combine both task halves
        data_conds = np.reshape(np.tile((np.array(['cond_%02d' % x for x in np.arange(no_RDM_conditions)])), (1,2)).transpose(),2*no_RDM_conditions)  
        # now prepare the data RDM file.
        sessions = np.concatenate((np.zeros(int(data_RDM_file['1'].shape[0])), np.ones(int(data_RDM_file['2'].shape[0]))))   
        # final data RDM file; cross correlated between task-halves.
        data_RDM = get_searchlight_RDMs(data_RDM_file_2d, centers, neighbors, data_conds, method='crosscorr', cv_descr=sessions)
        # save  so that I don't need to recompute
        with open(f"{results_dir}/data_RDM.pkl", 'wb') as file:
            pickle.dump(data_RDM, file)

 
    # Step 3: load and compute the model RDMs.

    # load the data files I created.
    data_dir = {}
    for model in models_I_want:
        data_dir[model]= np.load(os.path.join(RDM_dir, f"data{model}_{sub}_fmri_both_halves.npy")) 

    # step 3: create model RDMs.
    model_RDM_dir = {}
    RDM_my_model_dir = {}
    for model in data_dir:
        model_data = mc.analyse.analyse_MRI_behav.prepare_model_data(data_dir[model], no_RDM_conditions)
        model_RDM_dir[model] = rsr.calc_rdm(model_data, method='crosscorr', descriptor='conds', cv_descriptor='sessions')
        fig, ax, ret_vla = rsatoolbox.vis.show_rdm(model_RDM_dir[model])
        # then compute the location model.
        model_model = rsatoolbox.model.ModelFixed(f"{model}_only", model_RDM_dir[model])
        # Step 4: evaluate the model fit between model and data RDMs.
        RDM_my_model_dir[model] = Parallel(n_jobs=3)(delayed(mc.analyse.analyse_MRI_behav.evaluate_model)(model_model, d) for d in tqdm(data_RDM, desc=f"running GLM for all searchlights in {model}"))
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=RDM_my_model_dir[model], data_RDM_file=data_RDM, file_path = results_dir, file_name= f"{model}_betw_halves", mask=mask, number_regr = 0, ref_image_for_affine_path=ref_img)
       

    if RDM_version == '05': 
        
        clocks_midn_states_loc_ph_RDM = rsatoolbox.rdm.concat(model_RDM_dir['clocks'], model_RDM_dir['midnight'], model_RDM_dir['state'], model_RDM_dir['location'], model_RDM_dir['phase'])
        #clocks_midn_states_loc_ph_RDM = rsatoolbox.rdm.concat(clocks_RDM, midnight_RDM, state_RDM, loc_RDM, phase_RDM)
        clocks_midn_states_loc_ph_model = rsatoolbox.model.ModelWeighted('clocks_midn_states_RDM', clocks_midn_states_loc_ph_RDM)
        # the first is t, the second beta. [est.tvalues[1:], est.params[1:]]
        results_clocks_midn_states_loc_ph_model = Parallel(n_jobs=3)(delayed(mc.analyse.analyse_MRI_behav.evaluate_model)(clocks_midn_states_loc_ph_model, d) for d in tqdm(data_RDM, desc='running GLM for all searchlights in combo model 2'))
        
        
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "my_combo_clock_betw_halves", mask=mask, number_regr = 0, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "my_combo_midn_betw_halves", mask=mask, number_regr = 1, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "my_combo_state_betw_halves", mask=mask, number_regr = 2, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "my_combo_loc_betw_halves", mask=mask, number_regr = 3, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "my_combo_phase_betw_halves", mask=mask, number_regr = 4, ref_image_for_affine_path=ref_img)

    if RDM_version == '08': 
        #['reward_midnight_v2', 'reward_clocks_v2', 'state', 'task_prog']
        
        clocks_midn_states_loc_ph_RDM = rsatoolbox.rdm.concat(model_RDM_dir['clocks'], model_RDM_dir['midnight'], model_RDM_dir['state'], model_RDM_dir['location'], model_RDM_dir['phase'])
        #clocks_midn_states_loc_ph_RDM = rsatoolbox.rdm.concat(clocks_RDM, midnight_RDM, state_RDM, loc_RDM, phase_RDM)
        clocks_midn_states_loc_ph_model = rsatoolbox.model.ModelWeighted('clocks_midn_states_RDM', clocks_midn_states_loc_ph_RDM)
        # the first is t, the second beta. [est.tvalues[1:], est.params[1:]]
        results_clocks_midn_states_loc_ph_model = Parallel(n_jobs=3)(delayed(mc.analyse.analyse_MRI_behav.evaluate_model)(clocks_midn_states_loc_ph_model, d) for d in tqdm(data_RDM, desc='running GLM for all searchlights in combo model 2'))
        
        
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "my_combo_clock_betw_halves", mask=mask, number_regr = 0, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "my_combo_midn_betw_halves", mask=mask, number_regr = 1, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "my_combo_state_betw_halves", mask=mask, number_regr = 2, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "my_combo_loc_betw_halves", mask=mask, number_regr = 3, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "my_combo_phase_betw_halves", mask=mask, number_regr = 4, ref_image_for_affine_path=ref_img)

    if RDM_version == '09':
        # I want to have a combined 
        current_and_all_future_rew_RDM = rsatoolbox.rdm.concat(model_RDM_dir['reward_location'], model_RDM_dir['one_future_rew_loc'], model_RDM_dir['two_future_rew_loc'], model_RDM_dir['three_future_rew_loc'])
        current_and_all_future_rew_model = rsatoolbox.model.ModelWeighted('current_and_all_future_rew_RDM', current_and_all_future_rew_RDM)
        
        results_current_and_all_future_rew_model = Parallel(n_jobs=3)(delayed(mc.analyse.analyse_MRI_behav.evaluate_model)(current_and_all_future_rew_model, d) for d in tqdm(data_RDM, desc='running GLM for all searchlights in combo model'))
        
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_current_and_all_future_rew_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "curr_rew_combo_curr_and_all_future_rew", mask=mask, number_regr = 0, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_current_and_all_future_rew_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "one_fut_rew_combo_curr_and_all_future_rew", mask=mask, number_regr = 1, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_current_and_all_future_rew_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "two_fut_rew_combo_curr_and_all_future_rew", mask=mask, number_regr = 2, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_current_and_all_future_rew_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "three_fut_rew_combo_curr_and_all_future_rew", mask=mask, number_regr = 3, ref_image_for_affine_path=ref_img)
        
        # additionally do clocks vs. current rew location 
        clocks_curr_rew_RDM = rsatoolbox.rdm.concat(model_RDM_dir['reward_location'], model_RDM_dir['reward_clocks_v2'])
        clocks_curr_rew_RDM_model = rsatoolbox.model.ModelWeighted('clocks_curr_rew_RDM', clocks_curr_rew_RDM)
        
        results_current_and_all_future_rew_model = Parallel(n_jobs=3)(delayed(mc.analyse.analyse_MRI_behav.evaluate_model)(current_and_all_future_rew_model, d) for d in tqdm(data_RDM, desc='running GLM for all searchlights in combo model'))
        
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=clocks_curr_rew_RDM_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "curr_reward_location_combo_cl_mid", mask=mask, number_regr = 0, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=clocks_curr_rew_RDM_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "clocks_location_combo_cl_mid", mask=mask, number_regr = 1, ref_image_for_affine_path=ref_img)
        
    if RDM_version == '10':
        # I want to have a combined 
        current_and_all_future_rew_RDM = rsatoolbox.rdm.concat(model_RDM_dir['reward_location'], model_RDM_dir['one_future_rew_loc'], model_RDM_dir['two_future_rew_loc'])
        current_and_all_future_rew_model = rsatoolbox.model.ModelWeighted('current_and_all_future_rew_RDM', current_and_all_future_rew_RDM)
        
        results_current_and_all_future_rew_model = Parallel(n_jobs=3)(delayed(mc.analyse.analyse_MRI_behav.evaluate_model)(current_and_all_future_rew_model, d) for d in tqdm(data_RDM, desc='running GLM for all searchlights in combo model'))
        
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_current_and_all_future_rew_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "curr_rew_combo_curr_and_all_future_rew", mask=mask, number_regr = 0, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_current_and_all_future_rew_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "one_fut_rew_combo_curr_and_all_future_rew", mask=mask, number_regr = 1, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_current_and_all_future_rew_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "two_fut_rew_combo_curr_and_all_future_rew", mask=mask, number_regr = 2, ref_image_for_affine_path=ref_img)

        # additionally do clocks vs. current rew location 
        clocks_curr_rew_RDM = rsatoolbox.rdm.concat(model_RDM_dir['reward_location'], model_RDM_dir['reward_clocks_v2'])
        clocks_curr_rew_RDM_model = rsatoolbox.model.ModelWeighted('clocks_curr_rew_RDM', clocks_curr_rew_RDM)
        
        results_current_and_all_future_rew_model = Parallel(n_jobs=3)(delayed(mc.analyse.analyse_MRI_behav.evaluate_model)(current_and_all_future_rew_model, d) for d in tqdm(data_RDM, desc='running GLM for all searchlights in combo model'))
        
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=clocks_curr_rew_RDM_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "curr_reward_location_combo_cl_mid", mask=mask, number_regr = 0, ref_image_for_affine_path=ref_img)
        mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=clocks_curr_rew_RDM_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= "clocks_location_combo_cl_mid", mask=mask, number_regr = 1, ref_image_for_affine_path=ref_img)
