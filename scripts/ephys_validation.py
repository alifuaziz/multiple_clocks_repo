#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 15:58:33 2023

@author: Svenja Küchenhoff
This script validates my models with Mohamadys ephys data, making use of an RSA.
This is the plan:
 1.  Based on a location file (do you have something like that?),  model how the mouse was running: where was it for how long
 2.  Create a model RDM (time x time) based on my model using the coordinates from step (1)
 3.  Run an RSA where I compute data a RDM from all of your neuron time-series (all neurons across time, then correlate the time x time axes for the RDM)
 4.  Run a regression between my model RDM and your data RDM
 
 
 'Data is here: https://drive.google.com/drive/folders/1vJw8AVZmHQrUnvqkASUwAd4t549uKN6b'
 
 There are several neurons that have been recorded per task configuration. 
 
 For every recording, I have a task_configuration file.
 e.g. for the first day, the mice did 9 different tasks. 
 Therefore, I have 9 different files for neurons + locations for that day.
 Within the loc files, there are multiple runs of this same task configuration.
 Within the neuron files, there are multiple neurons and runs of this same task configuration.
 
"""

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import os, sys, pickle, time, re, csv
import mc
import pandas as pd
from datetime import datetime 
from matplotlib import cm
from itertools import product
import math 

# import pdb; pdb.set_trace()
take_raw_data = 1
take_bin_data = 0
average_bin = 0
regression_yes = 1
compare_two_tasks_yes = 1
continuous_model = 1


if average_bin == 1:
    # import pdb; pdb.set_trace()

    Data_folder='/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_recordings_200423/' 
    # start with a single mouse first.
    mouse_recday='me11_01122021_02122021'
    
    all_task_configs = np.load(Data_folder+'Task_data_'+mouse_recday+'.npy')
    no_task_configs = len(all_task_configs)
    locations = list()
    neurons = list()
    
    # load all data first.
    for session in range(0, no_task_configs):
        locations.append(np.load(Data_folder+'Location_'+mouse_recday+'_'+str(session)+'.npy'))
        neurons.append(np.load(Data_folder+'Neuron_'+mouse_recday+'_'+str(session)+'.npy'))
    
    for task_no, task_config in enumerate(all_task_configs):
        curr_neurons = neurons[task_no]
        # average over runs (2nd dim)
        # neurons x runs x timebins
        mean_neurons_per_task_start = np.mean(curr_neurons[:, 0:4, :], axis = 1)
        mean_neurons_per_task_learned = np.mean(curr_neurons[:, -5:-1, :], axis = 1)
        RSM_neurons = mc.simulation.RDMs.within_task_RDM(mean_neurons_per_task_start, plotting = True, titlestring = f"Start data RDM task {task_config}")
        RSM_neurons_two = mc.simulation.RDMs.within_task_RDM(mean_neurons_per_task_learned, plotting = True, titlestring = f"Learned data RDM task {task_config}")
        print(task_config)
        # create a concatenated version of these per task
        if task_no == 0:
            between_tasks_neurons = mean_neurons_per_task_learned.copy()
        elif task_no > 0:
            between_tasks_neurons = np.concatenate((between_tasks_neurons,mean_neurons_per_task_learned), axis = 1)
    
    # plot the neurons
    mc.simulation.predictions.plot_without_legends(between_tasks_neurons, titlestring= 'neuron average across tasks, 360 timebins', intervalline= 360)
    RSM_between_neurons = mc.simulation.RDMs.within_task_RDM(between_tasks_neurons, plotting=True, titlestring="RSM across tasks, 360 timebins, averaged across 4 last runs")
    
    # setting up a single phase-coded model.
    a = ([1, 0, 0],[0,1,0],[0,0,1])
    b = np.repeat(a,30, axis = 1)
    phase_model = np.tile(b, 4)
    
    RSM_phase = mc.simulation.RDMs.within_task_RDM(phase_model, plotting = True, titlestring = 'Phase RDM')
    
    # phase model with one short and one long phase
    # setting up a single phase-coded model.
    a = ([1, 0, 0],[0,1,1])
    b = np.repeat(a,30, axis = 1)
    phase_model_two = np.tile(b, 4)
    
    RSM_phase_two = mc.simulation.RDMs.within_task_RDM(phase_model_two, plotting = True, titlestring = 'Two phases RDM')
    
    
        


if take_bin_data == 1:
    # first try: just do the analysis across one run, for all neurons of this one task config.
    # later write a loop.
    # settings are:
    run_number = 18
    reward_config_number = 0
    
    ## SOME SETTINGS
    plot_paths_of_the_day = 0
    Data_folder='/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_recordings_200423/' 
    
    # start with a single mouse/ recording session first.
    mouse_recday='me11_05122021_06122021'
    session=0
    
    # load all data first.
    locations = np.load(Data_folder+'Location_'+mouse_recday+'_'+str(session)+'.npy')
    rewards_configs = np.load(Data_folder+'Task_data_'+ mouse_recday+'.npy')
    all_task_configs = np.load(Data_folder+'Task_data_'+mouse_recday+'.npy')
    data_neurons=np.load(Data_folder+'Neuron_'+mouse_recday+'_'+str(session)+'.npy')
    task_config = all_task_configs[session]
    
    # CHANGE THIS TO A LOOP/ GROUP ANAYSIS LATER
    # group analysis: what happens for
    #   doing the analysis across trials
    #   doing the analysis across runs (base my model on averaged locations)
    #   doing the analysis acorss runs (use more neural data)
    
    curr_task = locations[run_number]
    # run 0 *6*-7-4-5-2-1-0-*3*-4-7-8-5-4-1-*2*-5-*4*-7-6-3-4-1-2-5-8-7-6-3-0-1-4-3-6
    # run 20 seems optimal: *6*-*3*-0-1-*2*-5-*4*-7-*6*
    # 6-3-2-4
    curr_neurons = data_neurons[:,run_number,:]
    
    
    # 1.  Based on a location file (do you have something like that?),  model how the mouse was running: where was it for how long
    # Mohammady codes both for bridges and nodes. I am only considering nodes. 
    # Thus, first try what happend if I approximate the location to nodes.
    
    for i, field in enumerate(curr_task):
        if field > 9: 
            curr_task[i] = curr_task[i-1]
        if math.isnan(field):
            # keep the location bc of timebins
            curr_task[i] = curr_task[i-1]
                
    
    # important: fields need to be between 0 and 8, and keep them as integers!
    curr_task = [int((field_no-1)) for field_no in curr_task]
    task_config = [int((field_no-1)) for field_no in task_config]
    
    # build my models
    location_model = mc.simulation.predictions.set_location_ephys(curr_task, task_config, grid_size = 3, plotting = True)
    clock_model, midnight_model = mc.simulation.predictions.set_clocks_ephys(curr_task, task_config, grid_size = 3, phases = 3, plotting = True)
    
    # now create the model RDMs
    RSM_location = mc.simulation.RDMs.within_task_RDM(location_model, plotting = True, titlestring = 'Location RDM')
    RSM_clock = mc.simulation.RDMs.within_task_RDM(clock_model, plotting = True, titlestring = 'Clock RDM')
    RSM_midnight = mc.simulation.RDMs.within_task_RDM(midnight_model, plotting = True, titlestring = 'Midnight RDM')
    
    # now create the data RDM
    # I am wondering if this is correct, though - maybe should I select those neurons where I know fit my predictions?
    RSM_neurons = mc.simulation.RDMs.within_task_RDM(curr_neurons, plotting = True, titlestring = 'Data RDM')
    
    # Lastly, create a linear regression with RSM_loc,clock and midnight as regressors and data to be predicted
    reg_res = mc.simulation.RDMs.lin_reg_RDMs(RSM_neurons, regressor_one_matrix=RSM_location, regressor_two_matrix= RSM_clock, regressor_three_matrix= RSM_midnight)
    # NEXT
    # CHECK WY CLOCK RDM LOOKS LIKE IT DOES
    # CHECK HOW TO INTERPRET THESE RESULTS


if take_raw_data == 1:
    # Take the raw data instead.
    #####Raw data:
    
    
    """Neuron_raw arrays are matrices of shape neurons X bins
    each bin is the firing rate in a 25 ms timewindow
    
    Location_raw arrays are arrays of length equal to the number of bins for the 
    Neuron_raw matrix (may be 1 off)
    
    trialtimes arrays are times (in ms) of each state: the first four columns are 
    the start of each state the fifth column is the end of the last state (D)
    
    Note that you'll have to convert the trial times from ms to bin number to subset 
    the neuron and location arrays (i.e. divide by 25)"""
    
    # import pdb; pdb.set_trace()
    
    # first try: just do the analysis across one run, for all neurons of this one task config.
    # later write a loop.
    # settings are:
    run_number = 3
    reward_config_number = 0
    
    ## SOME SETTINGS
    plot_paths_of_the_day = 0
    Data_folder='/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_recordings_200423/' 
    
    # start with a single mouse/ recording session first.
    
    
    # load all data first.
    # start with a single mouse/ recording session first.
    mouse_recday='me11_05122021_06122021'
    #mouse_recday='ah04_01122021_02122021'
    
    session= 5
    locations_raw = np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(session)+'.npy')
    rewards_configs = np.load(Data_folder+'Task_data_'+ mouse_recday+'.npy')
    all_task_configs = np.load(Data_folder+'Task_data_'+mouse_recday+'.npy')
    data_neurons_raw = np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(session)+'.npy')
    timings_raw = np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(session)+'.npy')
    task_config = all_task_configs[session]
    
    
    # first convert trial times from ms to bin number to match neuron and location arrays 
    # (1 bin = 25ms)
    timings = timings_raw.copy()
    for r, row in enumerate(timings_raw):
        for c, element in enumerate(row):
            timings[r,c] = element/25


    # second, change locations and rewards to 0 and ignoring bridges
    locations = locations_raw.copy()
    for i, field in enumerate(locations_raw):
        if field > 9: 
            locations[i] = locations[i-1]
        if math.isnan(field):
            # keep the location bc of timebins
            locations[i] = locations[i-1]
                
    
    # important: fields need to be between 0 and 8, and keep them as integers!
    locations = [int((field_no-1)) for field_no in locations]
    task_config = [int((field_no-1)) for field_no in task_config]
    

    # I will do this differently. it's annoying to store runs of different lengths.
    # instead, I will have my subpaths, separately for every path
    # potentiall, I will want to take a mean across runs... let's start slow. 
    row = timings[4]
    
    # define current data
    # > potentially turn into a loop at some point
    trajectory = locations[row[0]:row[-1]]
    
    # ISSUE 21.04.23:
    # if there are ONLY 0 for one timestep, the np.corrcoef will output nan for that instance. Maybe better:
    # replace by super super low value
    curr_neurons = data_neurons_raw[:,row[0]:row[-1]]
    
    test_curr_neurons = curr_neurons.copy()
    for col_no, column in enumerate(test_curr_neurons.T):
        if np.all(column == 0):
            test_curr_neurons[:,col_no] = 0.00001
    
    
    # some pre-processing to create my models.
    # to count subpaths
    subpath_file = [locations[row[0]:row[1]+1], locations[row[1]+1:row[2]+1], locations[row[2]+1:row[3]+1], locations[row[3]+1:row[4]+1]]
    timings_curr_run = [(elem - row[0]) for elem in row]

    # to find out the step number per subpath
    step_number = [0,0,0,0] 
    for path_no, subpath in enumerate(subpath_file):
        for i, field in enumerate(subpath):
            if i == 0:
                count = 0
            elif field != subpath[i-1]:
                count+=1
        step_number[path_no] = count
       
    # mark where steps are made
    for field_no, field in enumerate(trajectory):
        if field_no == 0:
            index_make_step = [0]
        elif field != trajectory[field_no-1]:
            index_make_step.append(field_no)
            
            
    # plot an examplary neuron
    # ##Example Neuron activity
    # mouse_recday='me11_05122021_06122021'
    # session=0
    # neuron=0

    # data_neurons=np.load(Data_folder+'Neuron_'+mouse_recday+'_'+str(session)+'.npy')
    data_neuron = test_curr_neurons

    plt.matshow(data_neuron)
    for reward in timings_curr_run:
        plt.axvline(reward,color='red',ls='dashed')
    plt.show()

    # print(np.shape(data_neuron0))

    # data_neuron0    
    
    if continuous_model == 1:
        location_model, phase_model, state_model, midnight_model, clocks_model, phase_state_model = mc.simulation.predictions.set_continous_models_ephys(trajectory, timings_curr_run, index_make_step, step_number)

    
    elif continuous_model == 0:     
        location_model = mc.simulation.predictions.set_location_raw_ephys(trajectory, step_time = 1, grid_size=3, plotting = True, field_no_given= 1)
        midnight_model, clocks_model = mc.simulation.predictions.set_clocks_raw_ephys(trajectory, timings_curr_run, index_make_step, step_number, field_no_given= 1, plotting=True)
        phase_model = mc.simulation.predictions.set_phase_model_ephys(trajectory, timings_curr_run, index_make_step, step_number)
        
    
    if compare_two_tasks_yes == 1:
        session_two = 3
        locations_two_raw = np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(session_two)+'.npy')
        data_neurons_two_raw = np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(session_two)+'.npy')
        timings_two_raw = np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(session_two)+'.npy')
        task_two_config = all_task_configs[session_two]
    
        # first convert trial times from ms to bin number to match neuron and location arrays 
        # (1 bin = 25ms)
        timings_two = timings_two_raw.copy()
        for r, row in enumerate(timings_two_raw):
            for c, element in enumerate(row):
                timings_two[r,c] = element/25


        # second, change locations and rewards to 0 and ignoring bridges
        locations_two = locations_two_raw.copy()
        for i, field in enumerate(locations_two_raw):
            if field > 9: 
                locations_two[i] = locations_two[i-1]
            if math.isnan(field):
                # keep the location bc of timebins
                locations_two[i] = locations_two[i-1]
                    
        
        # important: fields need to be between 0 and 8, and keep them as integers!
        locations_two = [int((field_no-1)) for field_no in locations_two]
        task_two_config = [int((field_no-1)) for field_no in task_two_config]
        

        # I will do this differently. it's annoying to store runs of different lengths.
        # instead, I will have my subpaths, separately for every path
        # potentiall, I will want to take a mean across runs... let's start slow. 
        row = timings_two[4].copy()
        
        # define current data
        # > potentially turn into a loop at some point
        trajectory_two = locations_two[row[0]:row[-1]].copy()
        
        # ISSUE 21.04.23:
        # if there are ONLY 0 for one timestep, the np.corrcoef will output nan for that instance. Maybe better:
        # replace by super super low value
        curr_neurons_two = data_neurons_two_raw[:,row[0]:row[-1]].copy()
        
        test_curr_neurons_two = curr_neurons_two.copy()
        for col_no, column in enumerate(test_curr_neurons_two.T):
            if np.all(column == 0):
                test_curr_neurons_two[:,col_no] = 0.00001
        
        
        # some pre-processing to create my models.
        # to count subpaths
        subpath_file_two = [locations_two[row[0]:row[1]+1], locations_two[row[1]+1:row[2]+1], locations_two[row[2]+1:row[3]+1], locations_two[row[3]+1:row[4]+1]]
        timings_curr_run_two = [(elem - row[0]) for elem in row]

        # to find out the step number per subpath
        step_number = [0,0,0,0] 
        for path_no, subpath in enumerate(subpath_file):
            for i, field in enumerate(subpath):
                if i == 0:
                    count = 0
                elif field != subpath[i-1]:
                    count+=1
            step_number[path_no] = count
           
        # mark where steps are made
        for field_no, field in enumerate(trajectory_two):
            if field_no == 0:
                index_make_step = [0]
            elif field != trajectory_two[field_no-1]:
                index_make_step.append(field_no)
                
                
        # plot an examplary neuron
        # ##Example Neuron activity
        # mouse_recday='me11_05122021_06122021'
        # session=0
        # neuron=0

        # data_neurons=np.load(Data_folder+'Neuron_'+mouse_recday+'_'+str(session)+'.npy')
        data_neuron_two = test_curr_neurons_two.copy()

        plt.matshow(data_neuron_two)
        for reward in timings_curr_run_two:
            plt.axvline(reward,color='red',ls='dashed')
        plt.show()

        # print(np.shape(data_neuron0))

        # data_neuron0    
        if continuous_model == 1:
            location_model_two, phase_model_two, state_model_two, midnight_model_two, clocks_model_two, phase_state_model_two = mc.simulation.predictions.set_continous_models_ephys(trajectory, timings_curr_run, index_make_step, step_number)

        elif continuous_model == 0:        
            location_model_two = mc.simulation.predictions.set_location_raw_ephys(trajectory_two, step_time = 1, grid_size=3, plotting = True, field_no_given= 1)
            midnight_model_two, clocks_model_two = mc.simulation.predictions.set_clocks_raw_ephys(trajectory_two, timings_curr_run_two, index_make_step, step_number, field_no_given= 1, plotting=True)
            phase_model_two = mc.simulation.predictions.set_phase_model_ephys(trajectory_two, timings_curr_run_two, index_make_step, step_number)
        
        # concatenate task 1 and 2 and create RDMs to check similarity
        location_model_combined = np.concatenate((location_model, location_model_two), axis = 1)
        midnight_model_combined = np.concatenate((midnight_model, midnight_model_two), axis = 1)
        clocks_model_combined = np.concatenate((clocks_model, clocks_model_two), axis = 1)
        phase_model_combined = np.concatenate((phase_model, phase_model_two), axis = 1)
        neurons_combined = np.concatenate((data_neuron, data_neuron_two), axis = 1)
        
        # now create the RDMs
        RSM_location_combo = mc.simulation.RDMs.within_task_RDM(location_model_combined, plotting = True, titlestring = 'Location RSM combo')
        RSM_clock_combo = mc.simulation.RDMs.within_task_RDM(clocks_model_combined, plotting = True, titlestring = 'Clock RSM combo')
        RSM_midnight_combo = mc.simulation.RDMs.within_task_RDM(midnight_model_combined, plotting = True, titlestring = 'Midnight RSM combo')
        RSM_phase_combo = mc.simulation.RDMs.within_task_RDM(phase_model_combined, plotting = True, titlestring = 'Phase RSM combo')
        RSM_neurons_combo = mc.simulation.RDMs.within_task_RDM(neurons_combined, plotting = True, titlestring = 'Neurons RSM combo')
        

    
    
    if regression_yes == 1:
        # now create the regressors per run
        regs_phase_state_run = mc.simulation.predictions.create_regressors_per_state_phase_ephys(walked_path=trajectory, subpath_timings=timings_curr_run, step_no=step_number)
        # then use these regressors to generate a beta per neuron per run
        neurons_phase_state = mc.simulation.predictions.transform_data_to_betas(data_neuron, regs_phase_state_run)
        clock_phase_state = mc.simulation.predictions.transform_data_to_betas(clocks_model, regs_phase_state_run)
        midnight_phase_state= mc.simulation.predictions.transform_data_to_betas(midnight_model, regs_phase_state_run)
        location_phase_state = mc.simulation.predictions.transform_data_to_betas(location_model, regs_phase_state_run)
        phase_phase_state = mc.simulation.predictions.transform_data_to_betas(phase_model, regs_phase_state_run)
        
        
        # try how the stat_phase RDMS look like
        RSM_location_betas = mc.simulation.RDMs.within_task_RDM(location_phase_state, plotting = True, titlestring = 'Location phase*state dim RSM')
        RSM_clock_betas = mc.simulation.RDMs.within_task_RDM(clock_phase_state, plotting = True, titlestring = 'Clock phase*state dim RSM')
        RSM_midnight_betas = mc.simulation.RDMs.within_task_RDM(midnight_phase_state, plotting = True, titlestring = 'Midnight phase*state dim RSM')
        RSM_phase_betas = mc.simulation.RDMs.within_task_RDM(phase_phase_state, plotting = True, titlestring = 'Phase phase*state dim RSM')
        RSM_neurons_betas = mc.simulation.RDMs.within_task_RDM(neurons_phase_state, plotting = True, titlestring = 'Data phase*state dim RDM')
        
        
        
        # now create the model RDMs
        RSM_location = mc.simulation.RDMs.within_task_RDM(location_model, plotting = True, titlestring = 'Location RSM')
        RSM_clock = mc.simulation.RDMs.within_task_RDM(clocks_model, plotting = True, titlestring = 'Clock RSM')
        RSM_midnight = mc.simulation.RDMs.within_task_RDM(midnight_model, plotting = True, titlestring = 'Midnight RSM')
        RSM_phase = mc.simulation.RDMs.within_task_RDM(phase_model, plotting = True, titlestring = 'Phase RSM')
        
        # now create the data RDM
        # I am wondering if this is correct, though - maybe should I select those neurons where I know fit my predictions?
        RSM_neurons = mc.simulation.RDMs.within_task_RDM(test_curr_neurons, plotting = True, titlestring = 'Data RDM')
        
        # Lastly, create a linear regression with RSM_loc,clock and midnight as regressors and data to be predicted
        reg_res, scipy_regression_results = mc.simulation.RDMs.lin_reg_RDMs(RSM_neurons, regressor_one_matrix=RSM_clock, regressor_two_matrix= RSM_midnight, regressor_three_matrix= RSM_location, regressor_four_matrix= RSM_phase, t_val = 'yes')
        print(f" The beta for the clocks model is {reg_res.coef_[0]}, for the midnight model is {reg_res.coef_[1]}, for the location model is {reg_res.coef_[2]}, and for the phase model is {reg_res.coef_[3]}")
        print(scipy_regression_results.summary())
        
        # plot the variance in the off-diagonal elements of the clock as a histogram
        dimension = len(RSM_clock)
        diag_array = list(RSM_clock[np.tril_indices(dimension, -1)])
        fig, ax = plt.subplots()
        plt.hist(diag_array, bins = 10)
        plt.show()
    
    
    # 1. subject level.
    # for every mouse and every run, compute a GLM with my 3 regressors.
    # 2. compute contrasts:
    # I want to know: [0 0 1], [0 1 0], [1 0 0] and [-1 1 0], [0 -1 1], ....
    # (every regressor at its own, and the cotnrast between 2 betas (MRI: PEs) which is 
    # 1 minus the other)
    # take all of these values and average 
    #        1. across runs within one task config
    #       2. across task configs
    # Finally, you end up with 9 betas (MRI: COPEs) for every mouse (contrasts)
    # 3. Group level:
        # compute a random effects GLM for every of the contrasts, using
        # each mouse-beta as an input
    
    
    
    # next step: Stats! > group statistics?
    # > multiple runs? use the regressor model since there are different fields the mouse runs on? 
    # across tasks?? 
    # also check the correlation values, independent from the other regressors
    
    # before: within a mouse > fixed effects (+averaging betas) to compare runs 
    
    
    # last step: check ou FSL FEAT -> compare betas across mice with random effects if 
    # thats possible with only 8 mice, otherwise fixed effects 
    
    
    
    # potentially, also have a look at a second thing:
        # concatenate all trials across task cofngis
        # then reduce the size of the data to steps instead of ms
        # by using the step-regressors from the fMRI model (or the other way around)
    # afterwards, follow the same group-stats and contrasts
    # these contrasts are probably even more significant and will be more like my fmRI analyssi
    

#####################
## NOTEBOOK #########
#####################

'Data is here: https://drive.google.com/drive/folders/1vJw8AVZmHQrUnvqkASUwAd4t549uKN6b'

'''
####################
###Data structure###
####################


##Neuron npy files:
These contain the normalized firing rates of each neuron - spikes per frame (frame rate is 60 Hz)

npy files for each session
matrix with dimensions [neuron,trial,bin]

360 bins per trial, every state is 90 bins

##Location npy files:
These contain the location of the animal in each bin (should correspond exactly to the neuron bins)
locations1-9 are the 9 nodes

Then the remaining locations 10-21 are the bridges, coded in the "Edge_grid" array. just subtract 10 and that
gives you the index of the Edge_grid array which tells you which bridges are being referenced
e.g. an entry of 10 means index 0 which is array([1, 2]) (i.e. animal is at the the bridge between nodes 1 and 2)

##task_data
this is the task sequence used for that day (which nodes are rewarded in what order)

'''






#  # 1.  Based on a location file (do you have something like that?),  model how the mouse was running: where was it for how long
#  # Mohammady codes both for bridges and nodes. I am only considering nodes. 
#  # Thus, first try what happend if I approximate the location to nodes.
 
 
# # the location file per mouse and recording day is 
# ##Example Location array


# mouse_recday='me11_05122021_06122021'
# session=0
# location_=np.load(Data_folder+'Location_'+mouse_recday+'_'+str(session)+'.npy')

# print(np.shape(location_))
# location_[0]

# # the bridges are all numbers between 10-21
# # maybe I want to replace all bridges with the neighbouring place fields.

# curr_task_location = location_[0]
# for i, place in enumerate(curr_task_location):
#     if place > 9:
#         curr_task_location[i] = curr_task_location[i-1]
 

# # to know the reward locations, look here:
# ##example task array
# mouse_recday='me11_05122021_06122021'
# tasks=np.load(Data_folder+'Task_data_'+mouse_recday+'.npy')

# tasks

# # from the reward coordinates, I can infer how many steps between each reward were taken
# # goal is to create 'steps_per_walk' which is a list of e.g. 4 numbers that tells me how many steps between 2 rewards are taken.

# # then, write the numbers of the fields as coordinates. 
# # I can probably use some sort of reversed function as:
#     # first step is to do -1 (bc I have numbers 1 til 9 but I need 0 til 8)
#     # divide the number by 3 (grid size) into 2 whole numbers
#     # the result is x, the remainder is y.
#     # x = floor(number/grid_size)
#     # y = number - x
# #    y = step[1]
# #    x = step[0]
# #    anchor_field = x + y*size_grid

# for i in range(len(location_)):
#     curr_task = location_[i]
#     for field in curr_task:
#         print('yey')
        

# # locm, location_model = mc.simulation.predictions.set_location_by_time(walk, steps_per_walk, time_per_step, grid)
# # single_clock, midnight_matrix, clocks_model = mc.simulation.predictions.set_clocks_bytime(walk, steps_per_walk, time_per_step, grid)
    
 
# #############################################


# # THIS IS FROM MOHAMMADYS NOTEBOOK
# ##Recording days used
# '''These are pairs of days which were spike sorted together to give a total of 6 tasks: animals do 3 tasks a day'''

# Recording_days=np.load(Data_folder+'Recording_days_combined.npy')
# Recording_days


# ##example task array
# mouse_recday='me11_05122021_06122021'
# Tasks=np.load(Data_folder+'Task_data_'+mouse_recday+'.npy')

# Tasks


# #Edge grid
# '''use this to make sense of the location arrays (see data structure above)'''
# Edge_grid=np.load(Data_folder+'Edge_grid.npy')
# Edge_grid



# ##Example Location array

# mouse_recday='me11_05122021_06122021'
# session=0
# location_=np.load(Data_folder+'Location_'+mouse_recday+'_'+str(session)+'.npy')

# print(np.shape(location_))
# location_[0]


# ##Example Neuron activity
# mouse_recday='me11_05122021_06122021'
# session=0
# neuron=0

# data_neurons=np.load(Data_folder+'Neuron_'+mouse_recday+'_'+str(session)+'.npy')
# data_neuron0=data_neurons[neuron]

# plt.matshow(data_neuron0)
# for angle in np.arange(4)*90:
#     plt.axvline(angle,color='red',ls='dashed')
# plt.show()

# print(np.shape(data_neuron0))

# data_neuron0




# ######### GRAVEYARD #############
# # i dont think i need this ######

# if plot_paths_of_the_day == 1:
#     # have a look at how this walk looks like
#     # transform reward into field coordinates
#     reward_coords = []
#     for i, field in enumerate(task_config):
#         field_x_corr = (task_config[i]-1)//grid_size
#         field_y_corr = (task_config[i]-1) - field_x_corr*grid_size
#         reward_coords.append([field_x_corr, field_y_corr])
            
#     for run in range(0, len(location)):
#     # first delete all the fields where an animal is just slow.
#         curr_run_unique = [999]   # fill with 999 for now and delete in the end
#         curr_run = location[run]
#         # for i in range(len(curr_task)):
#         for i, field in enumerate(curr_run):
#             if field > 9:
#                 curr_run[i] = curr_run[i-1]                
#             if i > 0:
#                 if math.isnan(field) == False and curr_run[i] != curr_run_unique[-1]:
#                     curr_run_unique = np.append(curr_run_unique, curr_run[i])
        
#         curr_run_unique = curr_run_unique[1:] # get rid of the 99 again
        
#         # transform into field coordinates
#         unique_walk_coords = []
#         for i, field in enumerate(curr_run_unique):
#             field_x_corr = int((curr_run_unique[i]-1)//grid_size)
#             field_y_corr = int((curr_run_unique[i]-1) - field_x_corr*grid_size)
#             unique_walk_coords.append([field_x_corr, field_y_corr]) 
        
            
#         # not needed because the last function plots it all.    
#         # now plot, first plot the grid
#         # coord = [list(p) for p in product(range(grid_size), range(grid_size))]
#         # for curr_coords in reward_coords:    
#         #     plt.figure()
#         #     plt.axes()
#         #     cmap = cm.get_cmap('tab20b')
#         #     plt.scatter([x[0] for x in coord], [x[1] for x in coord], color =cmap(6), s=250)
#         #         # note that points[0:4] are my states: 
#         #             # reward_coords[1] = A - dark red
#         #             # reward_coords[2] = B - red
#         #             # reward_coords[3] = C - medium red
#         #             # reward_coords[4] = D - bright red  
#         # # then plot the reward
#         # for i, x in enumerate(reward_coords):
#         #     plt.scatter(x[0], x[1], color=cmap(i+11), s=250)
#         #     plt.yticks(list(range(grid_size)))
#         #     plt.xticks(list(range(grid_size)))
#         #     plt.grid(True)
            
#         # then plot where the mouse went.
#         mc.simulation.grid.plot_paths(reward_coords, unique_walk_coords)
    


# # now prepare the walked path and the reward coordinates for the script.
# # transform reward into field coordinates

    
# # the bridges are all numbers between 10-21
# # firstly, exchange all bridges for the fields the mouse was on before
# # secondly, create a 'walk' list made of x and y coordinates 
# for run_no in range(len(locations)):
#     # import pdb; pdb.set_trace() 
#     curr_task = locations[run_no]
#     curr_coords = []
#     for i, field in enumerate(curr_task):
#         if field > 9: 
#             curr_task[i] = curr_task[i-1]
#         if math.isnan(field):
#             # keep the location bc of timebins
#             curr_task[i] = curr_task[i-1]    
#         field_x_corr = int((curr_task[i]-1)//grid_size)
#         field_y_corr = int((curr_task[i]-1) - field_x_corr*grid_size)
#         curr_coords.append([field_x_corr, field_y_corr])
#     if run_no == 0:
#         curr_coords_np = np.array(curr_coords)
#         walk_coords = np.expand_dims(curr_coords_np, axis = 0)
#     if run_no > 0:
#         curr_coords_np = np.array(curr_coords)
#         walk_coords = np.concatenate([walk_coords, curr_coords_np.reshape([1, curr_coords_np.shape[0],curr_coords_np.shape[1]])], axis = 0)



    
# # thirdly, reverse the rewards_configs file into coordinates
# reward_coords = []
# for i, field in enumerate(task_config):
#     field_x_corr = (task_config[i]-1)//grid_size
#     field_y_corr = (task_config[i]-1) - field_x_corr*grid_size
#     reward_coords.append([field_x_corr, field_y_corr])    



# # 1.  Based on a location file (do you have something like that?),  model how the mouse was running: where was it for how long
# # Mohammady codes both for bridges and nodes. I am only considering nodes. 
# # Thus, first try what happend if I approximate the location to nodes.
# for run_no in range(len(locations)):
#     # import pdb; pdb.set_trace() 
#     curr_task = locations[run_no]
#     curr_coords = []
#     for i, field in enumerate(curr_task):
#         if field > 9: 
#             curr_task[i] = curr_task[i-1]
#         if math.isnan(field):
#             # keep the location bc of timebins
#             curr_task[i] = curr_task[i-1]

