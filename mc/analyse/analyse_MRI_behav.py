#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 13:14:44 2023
This file is to open and clean my behavioural variables.

@author: xpsy1114
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sps  
import statsmodels.api as sm
from nilearn.image import load_img
import os
import nibabel as nib
import statsmodels.api as sm
import shutil
from datetime import datetime
import warnings

import mc
import mc.analyse.analyse_MRI_behav as analyse_MRI_behav
import mc.replay_analysis.functions.utils as utils
import rsatoolbox.data as rsd
from rsatoolbox.rdm.calc import _build_rdms
from rsatoolbox.rdm import RDMs

def subpath_files(configs, subpath_after_steps, rew_list, rew_index, steps_subpath_alltasks):
    """
    Function construction the subpath_after_steps dictionary
    

    :returns steps_subpath_alltasks: dict of lists of lists for each task config
        - list of lists of the number of steps taken between each reward for each repeat of the task config
        i.e.[[4, 2, 3, 5],
             [4, 2, 3, 3],
             [4, 2, 3, 3],
             [4, 2, 3, 3],
             [6, 2, 3, 3]]
        is the number of steps taken between each reward (coin) for each of the 5 repeats of the task 'C1_forw'
    """


    for config in configs:
        rew_list[config] = [[int(value) for value in sub_list] for sub_list in rew_list[config][0:4]]
        # next step: create subpath files with rew_index and how many steps there are per subpath.
        
        # if task is completed
        if (len(subpath_after_steps[config])%4) == 0:
            for r in range(0, len(subpath_after_steps[config]), 4):
                subpath = subpath_after_steps[config][r:r+4]
                steps = [subpath[j] - subpath[j-1] for j in range(1,4)]
                if r == 0:
                    steps.insert(0, rew_index[config][r])
                if r > 0:
                    steps.insert(0, (subpath[0]- subpath_after_steps[config][r-1]))
                steps_subpath_alltasks[config].append(steps)
        # if task not completed
        elif (len(subpath_after_steps[config])%4) > 0:
            completed_tasks = len(subpath_after_steps[config])-(len(subpath_after_steps[config])%4)
            for r in range(0, completed_tasks, 4):
                subpath = subpath_after_steps[config][r:r+4]
                steps = [subpath[j] - subpath[j-1] for j in range(1,4)]
                if r == 0:
                    steps.insert(0, rew_index[config][r])
                if r > 0:
                    steps.insert(0, (subpath[0]- subpath_after_steps[config][r-1]))
                steps_subpath_alltasks[config].append(steps)
    
    return steps_subpath_alltasks


def extract_behaviour(file):
    """
    This function extracts the behavioural data from the csv file and returns a set of dictionaries corresponding to the different task parameters defined as follows


    :param file: str : path to the csv file containing the behavioural data

    :returns configs: list of set of routes that the participants took (e.g. ['C1_forw', 'D2_backw', 'A1_forw', 'C2_backw', 'E1_forw', 'E2_backw', 'D1_forw', 'B1_forw', 'B2_backw', 'A2_backw'])
        - Note: len(configs) = 10 (number of task configurations that each participant took through the routes)
    :returns rew_list: reward list
        - List of coin (reward) locations for each task config (e.g. [[0.0, 0.0], [2.0, 0.0], [0.0, 1.0], [2.0, 2.0], [0.0, 0.0], [2.0, 0.0], [0.0, 1.0], [2.0, 2.0], [0.0, 0.0], [2.0, 0.0], [0.0, 1.0], [2.0, 2.0], [0.0, 0.0], [2.0, 0.0], [0.0, 1.0], [2.0, 2.0], [0.0, 0.0], [2.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
        - This is a repeating pattern of 4 coins repeated 5 times for each task config i.e. list is 20 long
    :returns walked_path: dict of lists for each task config (e.g.) of the walked path of the agent 
        - Note: coordinate system used is 3x3 grid with 0,0 in the bottom LHS corner
        - This is alligned with the repeats of all the tasks concatenated
    :returns rew_index: dict of list for each task config
        - Represent the index (in the walked path) that the reward was obtained  
    :returns subpath_after_steps
        - Index represents the global index of the steps taken by the agent where reward was obtained
    :returns steps_subpath_alltasks: dict of lists for each task config (e.g.) of the number of steps taken between each reward
        - see the function subpath_files for more details on what is being returned
    :returns timings: dict of lists for each task config (e.g.) of the global times of the agents steps (w.r.t the start of the experiment)
        - list of global timings that each step was taken by the agent for each task config
    :returns regressors
        - ??
    """
    # load the two required excel sheets
    df = pd.read_csv(file)
    # the first row is empty so delete to get indices right
    df = df.iloc[1:].reset_index(drop=True)
    # fill gapss
    df['round_no'] = df['round_no'].fillna(method='ffill')
    df['task_config'] = df['task_config'].fillna(method='ffill')
    df['repeat'] = df['repeat'].fillna(method='ffill')
    # so that I cann differenatiate task config and direction
    df['config_type'] = df['task_config'] + '_' + df['type']
    
    # add columns whith field numbers 
    for index, row in df.iterrows():
        # current locations
        df.at[index, 'curr_loc_y_coord'] = analyse_MRI_behav.transform_coord(df.at[index,'curr_loc_y'], is_y=True, is_x = False)
        df.at[index, 'curr_loc_x_coord'] = analyse_MRI_behav.transform_coord(df.at[index,'curr_loc_x'], is_x=True, is_y = False)
        df.at[index, 'curr_rew_y_coord'] = analyse_MRI_behav.transform_coord(df.at[index,'curr_rew_y'], is_y=True, is_x = False)
        df.at[index, 'curr_rew_x_coord'] = analyse_MRI_behav.transform_coord(df.at[index,'curr_rew_x'], is_x=True, is_y = False)
        # and prepare the regressors: config type, state and reward/walking specific.
        if not pd.isna(row['state']):
            if not np.isnan(row['rew_loc_x']):
                df.at[index, 'time_bin_type'] =  df.at[index, 'config_type'] + '_' + df.at[index, 'state'] + '_reward'
            elif np.isnan(row['rew_loc_x']):
                df.at[index, 'time_bin_type'] = df.at[index, 'config_type'] + '_' + df.at[index, 'state'] + '_path'
    

    # create a dictionnary with all future regressors, to make sure the names are not messed up.
    time_bin_types = df['time_bin_type'].dropna().unique()
    regressors = {}
    for time_bin_type in time_bin_types:
        regressors[time_bin_type] = []
       

    configs = df['config_type'].dropna().unique()
    
    
    walked_path = {}
    timings = {}
    rew_list = {}
    rew_timing = {}
    rew_index = {}
    subpath_after_steps = {}
    steps_subpath_alltasks = {}
    for config in configs:
        walked_path[config] = []
        timings[config] = []
        rew_list[config] = []
        rew_timing[config] = []
        rew_index[config] = []
        subpath_after_steps[config] = []
        steps_subpath_alltasks[config] = []
    
    
    for index, row in df.iterrows():
        task_config = row['config_type']
        time_bin_type = row['time_bin_type']
        
        #iterate through the regression dictionary first
        for key in regressors.keys():
            # check if the key starts with the task_config value
            if key.startswith(task_config):
                if time_bin_type == key:
                    regressors[key].append(1)
                elif pd.notna(time_bin_type):
                    regressors[key].append(0) 

        # in case a new task has just started
        if not np.isnan(row['next_task']): 
            # first check if this is the first task of several repeats.
            if (index == 0) or (row['config_type'] != df.at[index -1, 'config_type']):
                timings[task_config].append(row['next_task'])
            else: # if it isnt, then take the reward start time from last rew D as start field.
                timings[task_config].append(df.at[index -1, 't_step_press_global'])
            walked_path[task_config].append([row['curr_loc_x_coord'], row['curr_loc_y_coord']])
        
        # if this is just a normal walking field
        elif not np.isnan(row['t_step_press_global']): # always except if this is reward D 
            # if its reward D, then it will be covered by the first if: if not np.isnan(row['next_task']): 
            timings[task_config].append(df.at[index - 1, 't_step_press_global'])  # Extract value from index-1
            walked_path[task_config].append([row['curr_loc_x_coord'], row['curr_loc_y_coord']])
       
        # next check if its a reward field
        if not np.isnan(row['rew_loc_x']): # if this is a reward field.
            # check if this is either at reward D(thus complete) or ignore interrupted trials
            # ignore these as they are not complete.
            if (index+2 < len(df)) or (row['state'] == 'D'):
                rew_timing[task_config].append(row['t_reward_start'])
                rew_list[task_config].append([row['curr_rew_x_coord'], row['curr_rew_y_coord']])
                subpath_after_steps[task_config].append(int(index-row['repeat']))  
                if row['state'] == 'D':
                    rew_index[task_config].append(len(walked_path[task_config])) #bc step has not been added yet
                    # if this is the last run of a task
                    if (index+2 < len(df)):
                        # first check if there are more tasks coming after, otherwise error
                        if (row['config_type'] != df.at[index +1, 'config_type']):
                            walked_path[task_config].append([row['curr_loc_x_coord'], row['curr_loc_y_coord']])
                            timings[task_config].append(df.at[index -1, 't_reward_start'])
                    else:
                        # however also add these fields if this is the very last reward!
                        if row['repeat'] == 4:
                            walked_path[task_config].append([row['curr_loc_x_coord'], row['curr_loc_y_coord']])
                            timings[task_config].append(df.at[index -1, 't_step_press_global'])
                            
                else:
                    rew_index[task_config].append(len(walked_path[task_config])-1) 
            else:
                continue

    return configs, rew_list, rew_index, walked_path, steps_subpath_alltasks, subpath_after_steps, timings, regressors


def models_I_want(RDM_version: str) -> list:
    """
    :param RDM_version: str : version of the RDMs to be used for the RSA analysis e.g. "01"

    :return models_I_want: list of models that I want to use for the RSA analysis as a list of strings
        - e.g. ['direction_presentation', 'execution_similarity', 'presentation_similarity']
    """

    if RDM_version in ['01', '01-1']: # 01 doesnt work yet! 
        models_I_want = ['direction_presentation', 'execution_similarity', 'presentation_similarity']
    # elif RDM_version in ['01-2']: 
    #     raise NotImplementedError(f"This version ({RDM_version}) is not implemented yet")
    elif RDM_version in ['01-2']:
        models_I_want = ['']
    elif RDM_version in ['02', '02-A']: #modelling paths + rewards, creating all possible models 
        models_I_want = ['location', 'phase', 'phase_state', 'state', 'task_prog', 'curr_rings_split_clock', 'one_fut_rings_split_clock', 'two_fut_rings_split_clock', 'three_fut_rings_split_clock', 'midnight', 'clocks']
    elif RDM_version in ['03', '03-A', '03-l', '03-e']: # modelling only rewards, splitting clocks within the same function
        models_I_want = ['location', 'phase', 'phase_state', 'state', 'task_prog', 'curr_rings_split_clock', 'one_fut_rings_split_clock', 'two_fut_rings_split_clock', 'three_fut_rings_split_clock', 'midnight_only-rew', 'clocks_only-rew']
    elif RDM_version in ['03-1', '03-2']:  # modelling only rewards, splitting clocks later in a different way - after the regression.
        models_I_want = ['location', 'phase', 'phase_state', 'state', 'task_prog', 'clocks_only-rew', 'midnight_only-rew', 'one_future_rew_loc' ,'two_future_rew_loc', 'three_future_rew_loc', 'curr-and-future-rew-locs']
    elif RDM_version in ['03-5', '03-5-A', '04-5', '04-5-A']:
        models_I_want = ['state']
    elif RDM_version in ['03-3']:  # modelling only rewards, splitting clocks later in a different way - after the regression; ignoring reward A
        models_I_want = ['location', 'phase', 'phase_state', 'state', 'task_prog', 'clocks_only-rew', 'midnight_only-rew', 'one_future_rew_loc' ,'two_future_rew_loc']
    elif RDM_version in ['03-99']:  # using 03-1 - reward locations and future rew model; but EVs are scrambled.
        models_I_want = ['location', 'phase', 'phase_state', 'state', 'task_prog', 'clocks_only-rew', 'midnight_only-rew', 'one_future_rew_loc' ,'two_future_rew_loc', 'three_future_rew_loc']
    elif RDM_version in ['03-999']:  # is debugging 2.0: using 03-1 - reward locations and future rew model; but the voxels are scrambled.
        models_I_want = ['location', 'phase', 'phase_state', 'state', 'task_prog', 'clocks_only-rew', 'midnight_only-rew', 'one_future_rew_loc' ,'two_future_rew_loc', 'three_future_rew_loc']
    elif RDM_version in ['04', '04-A']: # only paths. to see if the human brain represents also only those rings anchored at no-reward locations
        models_I_want = ['location', 'phase', 'phase_state', 'state', 'task_prog', 'curr_rings_split_clock', 'one_fut_rings_split_clock', 'two_fut_rings_split_clock', 'three_fut_rings_split_clock', 'midnight_no-rew', 'clocks_no-rew']
    else:
        raise ValueError(f"RDM_version {RDM_version} is not recognised.")

    # returns the list 
    return models_I_want

def preprocess_regression_version(REGRESSION_VERSION):
    """
    Function to preprocess the regression version string to ensure that it is in the correct format
    
    """

    if REGRESSION_VERSION in ['03-3', '03-4']:
        REGRESSION_VERSION = '03'
    if REGRESSION_VERSION in ['03-4-e']:
        REGRESSION_VERSION = '03-e'
    if REGRESSION_VERSION in ['03-4-l']:
        REGRESSION_VERSION = '03-l'
    if REGRESSION_VERSION in ['04-4']:
        REGRESSION_VERSION = '04'

    return REGRESSION_VERSION

def get_no_RDM_conditions(regression_version: str) -> int:
    """
    Returns the number of conditions that the RDM will be compared to for the RSA analaysis.
    This is, therefore, size of the RDM matrix 

    :param regression_version: str : version of the regression model used for the analysis
    :return: no_RDM_conditions (int) : number of conditions in the RDM model
    """
    if regression_version in ['01']:
        no_RDM_conditions = 20 # including all instruction periods. This is the number of separate conditions that will be compared to eachother
    elif regression_version in ['01-2']:
        # no_RDM_conditions = 10, we are treating all execution periods of the same set of rewards as the same condition.
        no_RDM_conditions = 10 
    elif regression_version in ['02', '02-e', '02-l']:
        no_RDM_conditions = 80 # including all paths and rewards
    elif regression_version in ['03', '04','03-99', '03-999', '03-9999', '03-l', '03-e']:
        no_RDM_conditions = 40 # only including rewards or only paths
    elif regression_version == '03-3': #excluding reward A
        no_RDM_conditions = 30
    elif regression_version in ['03-4', '04-4', '03-4-e', '03-4-l']: # only including tasks without double reward locs: A,C,D  and only rewards
        no_RDM_conditions = 24
    if regression_version in ['03-4', '04-4'] and RDM_version in ['03-5-A', '02-A', '03-A', '04-A', '04-5-A']: # only TASK A,C,D, only rewards B-C-D
        no_RDM_conditions = 18

    return no_RDM_conditions

def get_EV_dict(data_dir: str, regression_version: str) -> dict:
    """
    Function to read in the EVs for the GLM

    Returns a dictionary of the different events and their respective paths
    e.g. {'ev_press_EV_1': '/path/to/ev_press_EV_1.nii.gz', ...}
    
    """

    pe_path = f"{data_dir}/func/glm_{regression_version}.feat/stats"
    reading_in_EVs_dict = {}
    with open(f"{data_dir}/func/EVs_{regression_version}_pt01/task-to-EV.txt", 'r') as file:
        for line in file:
            index, name = line.strip().split(' ', 1)
            reading_in_EVs_dict[name] = os.path.join(pe_path, f"pe{int(index)+1}.nii.gz")
    return reading_in_EVs_dict


def move_files_to_subfolder(folder_path: str):
    # import pdb; pdb.set_trace()

    # List all files in the source folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    # Filter files that end with .nii.gz
    nii_gz_files = [f for f in files if f.endswith('.nii.gz')]

    if not nii_gz_files:
        print("No .nii.gz files found to move.")
    else:
        # Get today's date in the format YYYY-MM-DD
        today_date = datetime.today().strftime('%Y-%m-%d')
        subfolder_name = f"results_pre_{today_date}"
        
        # Create the sub-folder if it doesn't exist
        subfolder_path = os.path.join(folder_path, subfolder_name)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
        # Move .nii.gz files to the target sub-folder
        for file in nii_gz_files:
            shutil.move(os.path.join(folder_path, file), subfolder_path)
            print(f"Moved {file} to {subfolder_path}/")
         

def print_stuff(string_input: str):
    """
    print the string input
    """
    print(string_input)
    

def jitter(expected_step_no: int) -> np.array:


    # first randomly sample from a gamma distribution
    shape = 5.75 # this is what the mean subpath is supposed to be
    draw = np.random.standard_gamma(shape)
    
    # then make an array for each step + reward I expect to take
    step_size_maker = np.random.randint(1, expected_step_no + 4, size= expected_step_no + 1)
    
    # make the last one, the reward, twice as long as the average step
    ave_step = np.mean(step_size_maker)
    step_size_maker[-1] = ave_step*2
    
    # then multiply the fraction of all step sizes with the actual subpath length
    stepsizes = np.empty(expected_step_no + 1)
    for i in range(expected_step_no+ 1):
        stepsizes[i] = (step_size_maker[i]/ (sum(step_size_maker))) * draw
        
    # stepsizes [-1] will be reward length. if more steps than stepsizes[0:-2], randomly sample.
    
    return(stepsizes)



    # plotting how I draw the randomly jittered steps
    
    # # first randomly sample from a gamma distribution
    # # or from an exponantial
    # # then sample no of optimal steps random numbers 
    # # e.g. if 3 random numbers
    # # a/(a+b+c) * randomly sampled goal
    # # then 
    repeats = 10000
    shape, scale = 5.75, 1. # mean and width
    s = np.empty(repeats)
    for i in range(repeats):
        draw = np.random.standard_gamma(shape)
        while (draw < 3) or (draw > 15):
            draw = np.random.standard_gamma(shape)
        s[i] = draw
    
    step_no = 3
    step_size_maker = np.random.randint(1, step_no + 4, size= step_no + 1)
    ave_step = np.mean(step_size_maker)
    step_size_maker[-1] = ave_step*2
    
    # Find the index of the maximum value in the array
    # max_index = np.argmax(step_size_maker) 
    # Swap the maximum value with the last element
    # step_size_maker[max_index], step_size_maker[-1] = step_size_maker[-1], step_size_maker[max_index]

    stepsizes = np.empty(step_no + 1)
    for i in range(step_no+ 1):
        stepsizes[i] = (step_size_maker[i]/ (sum(step_size_maker))) * draw
        
    print (f'Step 1 = {stepsizes[0]} Step 2 = {stepsizes[1]} Step 3 = {stepsizes[1]}, rew = {stepsizes[-1]}, sum = {sum(stepsizes)}')
    
    plt.figure()
    count, bins, ignored = plt.hist(s, 50, density=True)
    y = bins**(shape-1) * ((np.exp(-bins/scale))/(sps.gamma(shape) * scale**shape))
    plt.plot(bins, y, linewidth=2, color='r')  
    plt.show()
    
    

def create_EV(onset, duration, magnitude, name, folder, TR_at_sec):
    """
    code snippet to create a regressor
    """
    if len(onset) > len(duration):
        onset = onset[:len(duration)]
        magnitude = magnitude[:len(duration)]
    elif len(duration) > len(onset):
        duration = onset[:len(onset)]
        magnitude = magnitude[:len(onset)]
    regressor_matrix = np.ones((len(magnitude),3))
    regressor_matrix[:,0] = [(time - TR_at_sec) for time in onset]
    regressor_matrix[:,1] = duration
    regressor_matrix[:,2] = magnitude
    # import pdb; pdb.set_trace()
    np.savetxt(str(folder) + 'ev_' + str(name) + '.txt', regressor_matrix, delimiter="    ", fmt='%f')
    return regressor_matrix


def transform_coord(coord, is_x = False, is_y = False):
    """
    Converts the coordinates (2d, is_x, is_y) of the agent to a 3x3 grid with 0,0 in the bottom LHS corner
    """

    if is_x:
        if coord == -0.21:
            return 0
        elif coord == 0:
            return 1
        elif coord == 0.21:
            return 2
    if is_y:
        if coord == -0.29:
            return 0
        elif coord == 0:
            return 1
        elif coord == 0.29:
            return 2
    # Add more conditions if needed
    else:
        return None



# use to check if the EV making went wrong
def check_for_nan(array):
    """
    Function to check if there are any NaNs in the array
    """

    # import pdb; pdb.set_trace()
    count = 0
    while np.isnan(array).any():
        print(f"Careful! There are Nans in {array}. Pausing script")
        # import pdb; pdb.set_trace()
        # try if this is sensible: delete the rows with the nans.
        array = array[0: (len(array)-1)]
        count = count + 1
    if count > 0:   
        print(f"deteleted {count} rows to avoid nans.")
    return count, array


def make_loc_EV(dataframe, x_coord, y_coord):
    # import pdb; pdb.set_trace()

    skip_next = False
    loc_dur = []
    loc_on = []
    loc_df = dataframe[(dataframe['curr_loc_x'] == x_coord) & (dataframe['curr_loc_y'] == y_coord)]
    #loc_one_on = loc_one['t_step_press_global'].to_list()
    # import pdb; pdb.set_trace()
    # try if this one works.
    # look at to check if its really the same task. For this, create a reward type 
    # column which allows to differentiate all trials
    loc_df['config_type'] = loc_df['task_config'] + '_' + loc_df['type']
    loc_df['config_type'] = loc_df['config_type'].fillna(method='ffill')
    for index, row in loc_df.iterrows():
        if index > 0: 
            if skip_next:
                skip_next = False
                continue
            # first case: a new repeat hast just started.
            if not np.isnan(row['next_task']): 
                start = dataframe.at[index, 'start_ABCD_screen']
                duration = dataframe.at[index, 't_step_press_global'] - start
                
            # second case: it is a reward. This can never be in 'next task', so else.
            elif not np.isnan(dataframe.at[index,'rew_loc_x']):
                if index+2 < len(dataframe): # only do this if this isn't the last row
                    # so here is a difference between reward A,B,C and D.
                    start = dataframe.at[index, 't_reward_start'] 
                    if row['state'] != 'D':
                        duration = dataframe.at[index + 1, 't_step_press_global'] - start
                    elif row['state'] == 'D':
                        # in case the next reward config is a different one
                        if row['config_type'] != dataframe.at[index+1, 'task_config']:
                            duration = dataframe.at[index, 't_reward_afterwait'] - start
                        # but if its just one repeat of many, it's more precise to take:
                        else:
                            duration = dataframe.at[index+1, 'start_ABCD_screen'] - start
                            # and if it was a 'D' within repeats, then skip the next row as it will be double otherwise!
                            skip_next = True
            # third case: its neither a new repeat, nor a reward.
            elif np.isnan(dataframe.at[index,'rew_loc_x']) and np.isnan(row['next_task']):
                start = dataframe.at[index-1, 't_step_press_global']
                duration = dataframe.at[index, 't_step_press_global'] - start
                
            loc_on.append(start)
            loc_dur.append(duration)
            
    loc_mag = np.ones(len(loc_on))
        
    return(loc_on, loc_dur, loc_mag)   
            

            
# FMRI ANALYSIS

# potentially delete this one
# def my_eval(model, data):
#       "Handle one voxel, copy the code that exists already for the neural data"
#       X = sm.add_constant(model.rdm.transpose());
#       Y = data.dissimilarities.transpose();
#       est = sm.OLS(Y, X).fit()
#       # import pdb; pdb.set_trace()
#       return est.tvalues[1:], est.params[1:], est.pvalues[1:]

# write a visualisation function for data RDMs.
def visualise_data_RDM(mni_x, mni_y, mni_z, data_RDM_file, mask):
    # import pdb; pdb.set_trace()
    x, y, z = mask.shape
    index_centre = np.ravel_multi_index((mni_x, mni_y, mni_z), (x,y,z))
    index_RDM = np.where(data_RDM_file.rdm_descriptors['voxel_index']==index_centre)[0]
    RDM_I_want= data_RDM_file[index_RDM].dissimilarities
    
    # matrix_40x40 = np.zeros((40, 40))

    # # Function to fill in the lower triangular part of the matrix
    # def fill_lower_triangular(matrix, data):
    #     indices = np.tril_indices_from(matrix)
    #     matrix[indices] = data
    
    # # Call the function to fill your matrix
    # fill_lower_triangular(matrix_40x40, RDM_I_want)

    # I believe that the RDMs are 80,93 - no actually its 40,93 for the half
    # well actually, an RDM should be no_conditions * no_conditions
    # why the hell is it 0,780??? this should be from 39x39 not 40x40???
    
    
    

def save_RSA_result(result_file, data_RDM_file, file_path, file_name, mask, number_regr, ref_image_for_affine_path):
    """
    Save the RSA results from the list of evaluations to a Nifti file for the t-values, beta-values and p-values
    
    """

    # Get the shape of the mask of the brain
    x, y, z = mask.shape
    # Load the reference image for the affine matrix
    ref_img = load_img(ref_image_for_affine_path)
    affine_matrix = ref_img.affine
    
    # Create the directory to for the results to be saved
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        
    # Create the array for the t-values
    t_result_brain = np.zeros([x*y*z])
    # ???
    t_result_brain[list(data_RDM_file.rdm_descriptors['voxel_index'])] = [vox[0][number_regr] for vox in result_file]
    # Reshape the t-values to the shape of the mask of the brain
    t_result_brain = t_result_brain.reshape([x,y,z])
    
    # Convert the t-values to a Nifti image
    t_result_brain_nifti = nib.Nifti1Image(t_result_brain, affine=affine_matrix)
    # define the file path and name for the t-values
    t_result_brain_file = f"{file_path}/{file_name}_t_val.nii.gz"
    # Save the t-values as a Nifti image
    nib.save(t_result_brain_nifti, t_result_brain_file)
    
    b_result_brain = np.zeros([x*y*z])
    b_result_brain[list(data_RDM_file.rdm_descriptors['voxel_index'])] = [vox[1][number_regr] for vox in result_file]
    b_result_brain = b_result_brain.reshape([x,y,z])
    
    b_result_brain_nifti = nib.Nifti1Image(b_result_brain, affine=affine_matrix)
    b_result_brain_file = f"{file_path}/{file_name}_beta.nii.gz"
    nib.save(b_result_brain_nifti, b_result_brain_file)
    
    p_result_brain = np.zeros([x*y*z])
    p_result_brain[list(data_RDM_file.rdm_descriptors['voxel_index'])] = [1 - vox[2][number_regr] for vox in result_file]
    p_result_brain = p_result_brain.reshape([x,y,z])
    
    p_result_brain_nifti = nib.Nifti1Image(p_result_brain, affine=affine_matrix)
    p_result_brain_file = f"{file_path}/{file_name}_p_val.nii.gz"
    nib.save(p_result_brain_nifti, p_result_brain_file)


def evaluate_model(model, data):
    # import pdb; pdb.set_trace()
    
    model= sm.add_constant(model.rdm.transpose());
    data = data.dissimilarities.transpose();
    
    # to filter out potential nans in the model part
    nan_filter = np.isnan(model).any(axis=1)
    filtered_model = model[~nan_filter]
    filtered_data = data[~nan_filter]
    
    # Instantiate Ordinary Least Squares model and fit it
    est = sm.OLS(filtered_data, filtered_model).fit()
    
    # Return the t-values, beta-values and p-values
    return est.tvalues[1:], est.params[1:], est.pvalues[1:]
    


def prepare_model_data(model_data: np.array,
                       no_Conditions: int, 
                       RDM_version: str):
    """
    
    Parameters
        model_data: N x no_Conditions matrix. 
            Each column is a condition and each row is a thoertical voxel that will be used to make an RDM matrix in the `rsr.calc_rdm` function
        no_Conditions: int : number of conditions in the model data
        RDM_version:
    
        
    Returns
        RSA_tb_model_data_object: rsd.Dataset : model data object for the RSA analysis. 
            The object contains the model data in the correct format for the RSA analysis.
    """
    # import pdb; pdb.set_trace()
    model_data = model_data.transpose()

    # Get the number of conditions 
    if RDM_version in ['01', '01-1', '01-2']:
        nCond = model_data.shape[0]
    else:
        nCond = model_data.shape[0]/2

    # Get the number of voxels
    nVox = model_data.shape[1]


    sessions = np.concatenate(
        (np.zeros(int(np.shape(model_data)[0]/2)),
          np.ones(int(np.shape(model_data)[0]/2)))
          )
    
    des = {'subj': 1}

    if RDM_version in ['01', '01-2']:
        conds = np.reshape(np.tile((np.array(['cond_%02d' % x for x in np.arange(nCond)])),(1)).transpose(),no_Conditions)
    else: 
        conds = np.reshape(np.tile((np.array(['cond_%02d' % x for x in np.arange(nCond)])), (1,2)).transpose(),no_Conditions*2)
    
    obs_des = {'conds': conds, 'sessions': sessions}

    chn_des = {'voxels': np.array(['voxel_' + str(x) for x in np.arange(nVox)])}
    
    # RSA data object
    RSA_tb_model_data_object = rsd.Dataset(measurements=model_data,
                                           descriptors=des,
                                           obs_descriptors=obs_des,
                                           channel_descriptors=chn_des)
    
    
    return RSA_tb_model_data_object


def analyse_pathlength_beh(df):
    # identify where the next task begins by iterating through the DataFrame 
    # and collecting the indices where the column is not empty
    index_next_task = []
    for index, row in df.iterrows():
        if not pd.isna(row['start_ABCD_screen']):
            index_next_task.append(index)
    
    # compute the task length for each task
    # careful! this only works if the task was completed.
    # also this isn't super precise since it doesn't actually show where they 
    # walked but where they were able to move away from reward
    for i, index in enumerate(index_next_task):
        if i+1 < len(index_next_task):
            df.at[index, 'task_length'] = df.at[index_next_task[i+1] - 1 , 't_reward_afterwait'] - df.at[index, 'start_ABCD_screen']   
            if 'type' in df.columns:
                df.at[index, 'type'] = df.at[index+ 1, 'type']
        elif i+1 == len(index_next_task):
            df.at[index, 'task_length'] = df.at[len(df)-1, 't_reward_afterwait'] - df.at[index, 'start_ABCD_screen'] 
                    
    # not sure why I included this... seems wrong.      
    # index_next_task = index_next_task[1:]
                    
    # identify where the next reward starts by iterating through the DataFrame 
    # and collecting the indices where the column is not empty
    index_next_reward = []
    for index, row in df.iterrows():
        if not pd.isna(row['t_reward_start']):
            index_next_reward.append(index)

    # Update 06.10.23: I don't think I need this anymore, I fixed it in the exp code
    # fill the missing last reward_delay columns.
    # they should be t_reward_afterwait-t_reward_start
    # take every 4th reward index to do so.
    #for i in range(3, len(index_next_reward), 4):
    #   df.at[index_next_reward[i], 'reward_delay'] = df.at[index_next_reward[i], 't_reward_afterwait'] - df.at[index_next_reward[i], 't_reward_start'] 
    
    # fill gaps in the round_no column
    df['round_no'] = df['round_no'].fillna(method='ffill')
    # do the same for the task_config 
    df['task_config'] = df['task_config'].fillna(method='ffill')
    # and create a reward type column which allows to differentiate all trials
    df['config_type'] = df['task_config'] + '_' + df['type']
    df['config_type'] = df['config_type'].fillna(method='ffill')
                
    
    # import pdb; pdb.set_trace()
    # create a new column in which you plot how long ever subpath takes (with rew)
    j = 0
    for i, task_index in enumerate(index_next_task):
        if task_index > 1:
            while (len(index_next_reward) > j) and (index_next_reward[j] < task_index):
                df.at[index_next_reward[j], 'cum_subpath_length_without_rew'] = df.at[index_next_reward[j], 't_step_press_curr_run'] + df.at[index_next_reward[j]-1, 'length_step'] 
                df.at[index_next_reward[j], 'cum_subpath_length_with_rew'] = df.at[index_next_reward[j], 't_step_press_curr_run'] + df.at[index_next_reward[j]-1, 'length_step'] + df.at[index_next_reward[j], 'reward_delay'] 
                j += 1
            df.at[task_index-1, 'cum_subpath_length_without_rew'] = df.at[index_next_task[i-1], 'task_length'] - df.at[task_index-1, 'reward_delay']
            df.at[task_index-1, 'cum_subpath_length_with_rew'] = df.at[index_next_task[i-1], 'task_length']
            # df.at[task_index-1, 't_step_press_curr_run'] + df.at[task_index-2, 'length_step'] + df.at[task_index-1, 'reward_delay'] 
        # for the next reward count backwards
        if task_index == index_next_task[-1]:
            for i in range(4,0, -1):
                df.at[index_next_reward[-i], 'cum_subpath_length_without_rew']= df.at[index_next_reward[-i], 't_step_press_curr_run'] + df.at[index_next_reward[-i]-1, 'length_step'] 
                df.at[index_next_reward[-i], 'cum_subpath_length_with_rew']= df.at[index_next_reward[-i], 't_step_press_curr_run'] + df.at[index_next_reward[-i]-1, 'length_step'] + df.at[index_next_reward[-i], 'reward_delay']

    states = ['A', 'B', 'C', 'D']*len(index_next_task)
    
    
    
    # then, write the not- cumulative columns.
    for i, reward_index in enumerate(index_next_reward):
        if i < len(states):
            df.at[reward_index, 'state'] = states[i]
        if i > 0:
            df.at[reward_index, 'subpath_length_without_rew'] = df.at[reward_index, 'cum_subpath_length_without_rew'] - df.at[index_next_reward[i-1], 'cum_subpath_length_with_rew']
            df.at[reward_index, 'subpath_length_with_rew'] = df.at[reward_index, 'cum_subpath_length_with_rew'] - df.at[index_next_reward[i-1], 'cum_subpath_length_with_rew']

    for i in range(0, len(index_next_reward), 4):
        df.at[index_next_reward[i], 'subpath_length_without_rew'] = df.at[index_next_reward[i], 'cum_subpath_length_without_rew'] 
        df.at[index_next_reward[i], 'subpath_length_with_rew'] = df.at[index_next_reward[i], 'cum_subpath_length_with_rew']

    
    #first reduce to only including those rows that have values for rewards.
    df_clean = df.dropna(subset = ['subpath_length_with_rew'])
    
    return(df, df_clean)



def similarity_of_tasks(reward_per_task_per_taskhalf_dict, RDM_VERSION):
    """
    
    Notes: 
    - `corrected_RSM_dict` is a dictionary that will store the corrected RSM for each model


    """
    # import pdb; pdb.set_trace() 
    assert RDM_VERSION in ['01'], f"The Regression version {RDM_VERSION} is not compatible with this function."
    
    # first, put the contents of the task-half dict into one.
    # def flatten_nested_dict(nested_dict):
    #     flattened_dict = {}
    #     for key, value in nested_dict.items():
    #         if isinstance(value, dict):  # If the value is a dictionary, extend the flat dictionary with its items
    #             flattened_dict.update(value)
    #         else:
    #             flattened_dict[key] = value
    #     return flattened_dict
    # rewards_experiment = flatten_nested_dict(reward_per_task_per_taskhalf_dict)
    
    rewards_experiment = utils.flatten_nested_dict(reward_per_task_per_taskhalf_dict)
    
    all_rewards = []
    all_names = []
    #  make sure that the dictionary is alphabetically sorted.
    for task in sorted(rewards_experiment.keys()):
        all_rewards.append(rewards_experiment[task])
        all_names.append(task)
    
    
    # create 3 binary RDMs:
        # 1. those that are backwards vs those that are forwards.
        # 2. those that are executed in the same order.
        # 3. those that are presented in the same order.
    
    
    # first, all those that are presented in a forward or backward way are equal.
    # direction_presentation = np.zeros((len(all_names), 2)) this will yield -1 and 1.
    direction_presentation = np.zeros((len(all_rewards), len(all_rewards)*4)) # this is -0.012658227848101285 and 1
    for i, task_name in enumerate(all_names):
        if task_name.endswith('forw'):
            direction_presentation[i, 0] = 1
        elif task_name.endswith('backw'):
            direction_presentation[i, 1] = 1
    
    # second, all those that are executed in the same order are the same.
    execution_similarity = np.zeros((len(all_rewards), len(all_rewards)*4)) # this is -0.012658227848101285 and 1
    for i in range(len(all_rewards)):
        for j in range(len(all_rewards)):
            if all_rewards[i] == all_rewards[j]:
                execution_similarity[i, j] = 1
            
            
    # third, all those that are presented in the same order are the same.
    # careful, this also changes all_rewards!! do this last.
    presented_rewards = all_rewards.copy()
    for i, task_name in enumerate(all_names):
        if task_name.endswith('backw'):
            presented_rewards[i].reverse()

    # Creating the presentation similarity matrix 
    presentation_similarity = np.zeros((len(all_rewards), len(all_rewards)*4)) # this is -0.012658227848101285 and 1
    for i in range(len(presented_rewards)):
        for j in range(len(presented_rewards)):
            if presented_rewards[i] == presented_rewards[j]:
                presentation_similarity[i, j] = 1
    
    # initialise corrected_RSM_dict
    corrected_RSM_dict = {}
    # for model in all_models_dict.keys(): # the list of models that want to be analyised, note that that "all_models_dict" is not currently passed to the function
    for model in ['direction_presentation', 'execution_similarity', 'presentation_similarity']:
        np.corrcoef(presentation_similarity[:, :10])
        # corrected_model = (presentation_similarity[:, :10] + np.transpose(presentation_similarity[:, :10]))/2
        corrected_model = (presentation_similarity[:, :10] + presentation_similarity[:, :10])/2
        
        corrected_RSM_dict[model] = corrected_model[0:int(len(corrected_model)/2), int(len(corrected_model)/2):]
    
        
    # to create the right format, split this into two task halves again
    # import pdb; pdb.set_trace() 
    models_between_tasks = {'execution_similarity': {key: "" for key in ['1', '2']},
                            'presentation_similarity': {key: "" for key in ['1', '2']},
                            'direction_presentation': {key: "" for key in ['1', '2']}}
    
    models_between_tasks['execution_similarity']['1'] = execution_similarity[:10].T
    models_between_tasks['execution_similarity']['2'] = execution_similarity[10:20].T
    
    models_between_tasks['presentation_similarity']['1'] = presentation_similarity[:10].T
    models_between_tasks['presentation_similarity']['2'] = presentation_similarity[10:20].T
    
    models_between_tasks['direction_presentation']['1'] = direction_presentation[:10].T
    models_between_tasks['direction_presentation']['2'] = direction_presentation[10:20].T

    # import pdb; pdb.set_trace()   
    # CONTINUE HERE!!! THE  PRESENT SIM ISNT QUITE RIGHT YET!
    return models_between_tasks

def auto_corr_RSM_dict(RSM_dict_betw_TH: dict) -> dict:
        """
        Create a corrected RSM dictionary by averaging the lower or upper triangles of the RSMs
        # then average the lower triangle and the top triangle of this nCond x nCond matrix, 
        # by adding it to its transpose, dividing by 2, and taking only the lower or 
        # upper triangle of the result.   

        Creates the correlation between task halves matrix  
        removes temporal correlation by averaging the off diagronal of part 1 and part 2 of the whole RSM

        Parameters

        Returns
            Corrected RSM dictionary. This entry contains the corrected RSM for each model

        """
        corrected_RSM_dict = {}
        for model in RSM_dict_betw_TH:
            # import pdb; pdb.set_trace()
            corrected_model = (RSM_dict_betw_TH[model] + np.transpose(RSM_dict_betw_TH[model]))/2
            corrected_RSM_dict[model] = corrected_model[0:int(len(corrected_model)/2), int(len(corrected_model)/2):]
        return corrected_RSM_dict


def auto_corr_RDM_obj(
        RDM_obj: RDMs
    ) -> RDMs:
    """
    Removes autocorrelation from the RDM object by averaging the lower or upper triangles of the RDMs
    """
    
    # Create a copy of the RDM object
    corrected_RDM_obj = RDM_obj.copy()

    # Get the RDM from the RDM object
    RDM_array = corrected_RDM_obj.dissimilarities

    corrected_RDM = (RDM_array + np.transpose(RDM_array)) / 2
    corrected_RDM = corrected_RDM[0:int(len(corrected_RDM)/2), int(len(corrected_RDM)/2):]

    # Update the RDM in the RDM object
    corrected_RDM_obj.dissimilarities = corrected_RDM

    # Update the labels of the RDM object
    

    # Return the copy of the RDM object with the corrected RDM
    return corrected_RDM_obj


def plot_trajectories(data):
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, len(data) * 2))  # Adjusted figure size for better fit

    # Custom colors for each value in the 3x3 grid
    colors = {
        0: '#003366',  # Dark blue
        1: '#005577',  # Less dark blue
        2: '#007799',  # Light blue
        3: '#006666',  # Dark turquoise
        4: '#008888',  # Less dark turquoise
        5: '#00AAAA',  # Light turquoise
        6: '#005555',  # Very dark turquoise
        7: '#007777',  # Darker turquoise
        8: '#009999',  # Bright turquoise
    }

    for half_id, configs in data.items():
        for config_id, runs in configs.items():
            # Set up the figure and axis for each task-half and config
            fig, ax = plt.subplots(figsize=(10, len(runs) * 2))
            plt.title(f'{config_id}, Task Half {half_id}', fontsize=16)

            for i, (run_id, values) in enumerate(runs.items()):
                y = len(runs) - i - 1  # Calculate the y-position for the current run
                
                # Plot each number as a circle with the respective color
                for x, value in enumerate(values):
                    ax.add_patch(plt.Circle((x, y), 0.4, color=colors[value]))
                    ax.text(x, y, str(value), color='white', ha='center', va='center', fontsize=12)

            ax.set_xlim(-1, len(values))
            ax.set_ylim(-1, len(runs))
            ax.set_aspect('equal')
            ax.axis('off')  # Turn off the axis
            plt.show()




if __name__ == "__main__":
    print("This is the analyse_MRI_behav.py script.")