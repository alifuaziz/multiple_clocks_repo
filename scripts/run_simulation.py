#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 17:05:51 2023

@author: Svenja Kuechenhoff

this script calls on several simulation functions to eventually create RDMs
and check if predictions of location neuron activations are sufficiently 
distinct from phase clock neuron activation patterns.

It is mainly thought to check if the simulations are correct. It therefore 
creates loads of figures for task configuration, paths, resulting matrices,
and neuron activations.

This script only creates the similarity between clocks and locations for one task
configuration. 

"""

# %reset -f

import mc
import pandas as pd
from matplotlib import pyplot as plt
import statistics as stats

## SETTINGS
section_oneone = 1 # Create the task
section_onetwo = 0 # Create a distribution of most common pathlengths
section_twoone = 1 # Setting the Clocks and Location Matrix. 
section_twotwo = 1 # Setting the Clocks but in 'real time'
section_three = 0 # Create 'neuron plots'
section_four = 0 # Create RDMs.


## Section 1.1
## Create the task
##
if section_oneone == 1:
    reward_coords = mc.simulation.grid.create_grid(plot = True)
    reshaped_visited_fields, all_stepnums = mc.simulation.grid.walk_paths(reward_coords, plotting = True)

############## 

## Section 1.2
## Create a distribution of most common pathlengths
##
if section_onetwo == 1:
    pathlenghts = list()

    for i in range(1,10000):
        reward_coords = mc.simulation.grid.create_grid(plot = False)
        reshaped_visited_fields, all_stepnums = mc.simulation.grid.walk_paths(reward_coords, plotting = False)
        pathlenghts.append(len(reshaped_visited_fields))
    plt.figure()    
    plt.hist(pathlenghts) 
    print(stats.mean(pathlenghts))
    print(stats.mode(pathlenghts))
    print(stats.median(pathlenghts))
    print(max(pathlenghts))
############## 


 
## Section 2.1
## Setting the Clocks and Location Matrix. 
##

# now create the two matrices, print them and plot them. 
if section_twoone == 1:
    first_clocks_matrix, total_steps  = mc.simulation.predictions.set_clocks(reshaped_visited_fields, all_stepnums, 3)           
    location_matrix, total_steps = mc.simulation.predictions.set_location_matrix(reshaped_visited_fields, all_stepnums, 3, 0) 
     
    print(first_clocks_matrix)
    print(location_matrix)
     
    mc.simulation.predictions.plotlocation(location_matrix)
    mc.simulation.predictions.plotclocks(first_clocks_matrix)
    
    one_clock = first_clocks_matrix[144:156,:]
    mc.simulation.predictions.plot_one_clock(one_clock)

# #########################


## Section 2.2
## Setting the Clocks but in 'real time'
##
if section_twotwo == 1:    
    reward_coords = mc.simulation.grid.create_grid(plot = True)
    reshaped_visited_fields, all_stepnums = mc.simulation.grid.walk_paths(reward_coords, plotting = True)
    
    clocksm, neuroncl, clocks_over_time = mc.simulation.predictions.set_clocks_bytime_one_neurone(reshaped_visited_fields, all_stepnums, 3, 2)
    clocksm
    neuroncl
    clocks_over_time
    
# see if I can plot the clocks_over_time
 # plotclock_pertime(clocks_matrix, step_time, all_stepnums):
    mc.simulation.predictions.plotclock_pertime(clocks_over_time, 2, all_stepnums)

#clocks_matrix, dummy_matrix, clock_neurons, clocks_per_step
# #########################


## Section 3. 
## Create 'neuron plots'
##

if section_three == 1:
    # this is the activity of one neuron based on a single run.
    plt_neurontwo = first_clocks_matrix[146,:]
    randomneuronsin = pd.DataFrame({'value': plt_neurontwo, #these will be averages
                          'bearing': range(0, 360, 30),
                          'phases': ['4. reward', '1. early', '1. late', '1. reward', '2. early', '2. late', '2. reward', '3. early', '3. late', '3. reward', '4. early', '4. late']})   
    mc.simulation.predictions.plot_neurons(randomneuronsin)
     
    # to plot the average activity of neuron across many runs, create data of many runs
    loops = 1000
    matrixtype = 'location'
    first_average_loc = mc.simulation.predictions.many_configs_loop(loops, matrixtype)
    matrixtypetwo = 'clocks'
    first_average_clock = mc.simulation.predictions.many_configs_loop(loops, matrixtypetwo)
    # 0,36,72,108,144,180,216,252,288 are respective field-anchors. 144 might be most interesting
    
    # PLOT MATRICES
    # Location
    mc.simulation.predictions.plotlocation(first_average_loc)
    # Clocks
    mc.simulation.predictions.plotclocks(first_average_clock)
    # Clocks, one anchor = 3 clocks
    anchor_five = first_clocks_matrix[144:179,:]
    # Clocks, one clock = 12 neurons
    mc.simulation.predictions.plot_one_anchor_all_clocks(anchor_five)
    
    one_clock = first_clocks_matrix[144:156,:]
    mc.simulation.predictions.plot_one_clock(one_clock)
    
    # ALIGN BY PHASE - PLOT NEURONS (Polar plots)
    # plot two CLOCK neuron examples, based on average.
    # example 1
    plt_neuron = first_average_clock[145,:]
    randomneuron = pd.DataFrame({'value': plt_neuron, #these will be averages
                          'bearing': range(0, 360, 30),
                          'phases': ['4. reward', '1. early', '1. late', '1. reward', '2. early', '2. late', '2. reward', '3. early', '3. late', '3. reward', '4. early', '4. late']})   
    mc.simulation.predictions.plot_neurons(randomneuron)
    
    #example 2
    plt_neurontwo = first_average_clock[146,:]
    randomneurontwo = pd.DataFrame({'value': plt_neurontwo, #these will be averages
                          'bearing': range(0, 360, 30),
                          'phases': ['4. reward', '1. early', '1. late', '1. reward', '2. early', '2. late', '2. reward', '3. early', '3. late', '3. reward', '4. early', '4. late']})   
    mc.simulation.predictions.plot_neurons(randomneurontwo)
    
    # plot LOCATION neuron average like this
    loc_neuron = first_average_loc[2,:]
    plt_loc_neuron = pd.DataFrame({'value': loc_neuron, #these will be averages
                          'bearing': range(0, 360, 30),
                          'phases': ['4. reward', '1. early', '1. late', '1. reward', '2. early', '2. late', '2. reward', '3. early', '3. late', '3. reward', '4. early', '4. late']})   
    mc.simulation.predictions.plot_neurons(plt_loc_neuron)
    

#########################
## Section 4. 
## Create RDMs.
##

if section_four == 1:
    phases = ['first_early', 'first_late', 'first_reward', 'scnd_early', 'scnd_late', 'scnd_reward', 'third_early', 'third_late', 'third_reward', 'fourth_early', 'fourth_late', 'fourth_reward']
    
    loc_RSM = mc.simulation.RDMs.within_task_RDM(location_matrix, phases)
    clock_RSM = mc.simulation.RDMs.within_task_RDM(first_clocks_matrix, phases)
    similarity = mc.simulation.RDMs.corr_matrices(loc_RSM, clock_RSM)
    
    print(similarity)



# # next steps: create RSMs across tasks: e.g. 3 different task configurations.


# # this will be phase-vectors corr with phase vectors. 
# # do the same with location and clock neurons.
# # first RDM: within a task.
# # 



# ###########################
# # Section 5. Correlate RDMs.
# I did the rest in another script....


      
