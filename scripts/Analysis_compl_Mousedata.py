#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 20:55:36 2023

@author: Svenja Küchenhoff
this script runs the entire analysis of Mohamadys data
"""

import numpy as np
import mc
import matplotlib.pyplot as plt
#import joypy
#from matplotlib import cm
#import plotly.figure_factory as ff
#import plotly.io as pio
#pio.renderers.default = "browser"
#import scipy.stats 
import os
import pickle

#
save = True
load_old = False

# Part 1: load data
mouse_a, mouse_b, mouse_c, mouse_d, mouse_e, mouse_f, mouse_g, mouse_h = mc.simulation.analyse_ephys.load_ephys_data(Data_folder = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_recordings_200423/')

# defining contrasts.
contrast_matrix = ((1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1), (1,-1,0,0), (1, 0,-1,0), (1,0,0,-1), (0,1,-1,0), (0,1,0,-1), (0,0,1,-1))


if save: 
    # to save the data later
    date = np.datetime64('today')
    out_path = f"/Users/xpsy1114/Documents/projects/multiple_clocks/output/{date}"
    res_path = "/Users/xpsy1114/Documents/projects/multiple_clocks/output/2023-06-14"
    if os.path.isdir(out_path) == False:
        os.mkdir(out_path)



if load_old: 
    # if I want to load old stuff, use this:
        # USE THIS TOMORROW!
    mice_recdays = ['mouse_a', 'mouse_b', 'mouse_c', 'mouse_d']
    # eval(f"{mice_recdays[0]}['recday']")
    clock_results = {}
    midnight_results = {}
    all_results = {}
    res_path = "/Users/xpsy1114/Documents/projects/multiple_clocks/output/2023-06-14"
    for i, mouse_index in enumerate(mice_recdays): 
        mouse_recday = eval(f"{mouse_index}['recday']")
        clock_file_name = f"{mouse_recday}_clock_dic"
        midn_file_name = f"{mouse_recday}_clock_dic"
        results_file_name = f"{mouse_recday}_res_dic"
        with open(os.path.join(res_path,clock_file_name), 'rb') as f:
            clock_results[clock_file_name] = pickle.load(f)
        with open(os.path.join(res_path,midn_file_name), 'rb') as f:
            midnight_results[midn_file_name] = pickle.load(f)
        with open(os.path.join(res_path,results_file_name), 'rb') as f:
            all_results[results_file_name] = pickle.load(f)
        
        


# my goal is to show that my model can predict Mohamadys data.
# 13.06.2023
# first, figure out which model is good. 
# compare continous vs normal
# compare different amount of cells
# compare different binnings.



# THIS IS A FREAKING MESS.
# CLEAN THIS UP AT SOME POINT....
# ok I think before I do the big thingy, I first have to go back to only running one single one.

# mouse a
mouse_a_clean =  {}
mouse_a_clean["cells"] = mouse_a["cells"].copy()
mouse_a_clean["recday"] = mouse_a["recday"]
mouse_a_clean["rewards_configs"], mouse_a_clean["locations"], mouse_a_clean["neurons"], mouse_a_clean["timings"] = mc.simulation.analyse_ephys.clean_ephys_data(mouse_a["rewards_configs"], mouse_a["locations"], mouse_a["neurons"], mouse_a["timings"], mouse_a_clean["recday"])

#a_reg_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(mouse_a_clean["rewards_configs"], mouse_a_clean["locations"], mouse_a_clean["neurons"], mouse_a_clean["timings"], mouse_a_clean["recday"], plotting = False, continuous = True, no_bins_per_state = 10, number_phase_neurons = 3)

# whole dataset
essi

# define a mask for clock and midnight neurons, respectively
clock_neurons = []
midnight_neurons = []
clock_n_mask = mouse_a_clean["cells"][:,-1]
midnight_n_mask = np.logical_and(clock_n_mask, mouse_a_clean["cells"][:,-2])

for neurons in mouse_a_clean["neurons"]:
    clock_neurons.append(neurons[clock_n_mask, :])
    midnight_neurons.append(neurons[midnight_n_mask, :])


print(f"Neuron number of recday {mouse_a['recday']} is {len(mouse_a_clean['neurons'][0])}, Midnight neuron number is {len(midnight_neurons[0])} and Clock neurons are {len(clock_neurons[0])}")    
           
a_midn_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(mouse_a_clean["rewards_configs"], mouse_a_clean["locations"], midnight_neurons, mouse_a_clean["timings"], mouse_a_clean["recday"], plotting = False, continuous = True, no_bins_per_state = 10, number_phase_neurons = 3, mask_within= True, split_by_phase = True)
a_clocks_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(mouse_a_clean["rewards_configs"], mouse_a_clean["locations"], clock_neurons, mouse_a_clean["timings"], mouse_a_clean["recday"], plotting = False, continuous = True, no_bins_per_state = 10, number_phase_neurons = 3, mask_within= True, split_by_phase = True)

if save:
    with open(os.path.join(out_path,f"{mouse_a_clean['recday']}_res_dic"), 'wb') as f:
        pickle.dump(a_reg_result_dict, f)
    
    with open(os.path.join(out_path,f"{mouse_a_clean['recday']}_midn_dic"), 'wb') as f:
        pickle.dump(a_midn_result_dict, f)
    
    with open(os.path.join(out_path,f"{mouse_a_clean['recday']}_clock_dic"), 'wb') as f:
        pickle.dump(a_clocks_result_dict, f)


    

# mouse b
mouse_b_clean =  {}
mouse_b_clean["cells"] = mouse_b["cells"].copy()
mouse_b_clean["recday"] = mouse_b["recday"]
mouse_b_clean["rewards_configs"], mouse_b_clean["locations"], mouse_b_clean["neurons"], mouse_b_clean["timings"] = mc.simulation.analyse_ephys.clean_ephys_data(mouse_b["rewards_configs"], mouse_b["locations"], mouse_b["neurons"], mouse_b["timings"], mouse_b_clean["recday"])

#b_reg_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(mouse_b_clean["rewards_configs"], mouse_b_clean["locations"], mouse_b_clean["neurons"], mouse_b_clean["timings"], mouse_b_clean["recday"], plotting = False, continuous = True, no_bins_per_state = 10, number_phase_neurons = 3)
# whole dataset
b_reg_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(mouse_b["rewards_configs"], mouse_b["locations"], mouse_b["neurons"], mouse_b["timings"], mouse_b["recday"], plotting = False, continuous = True, no_bins_per_state = 10, number_phase_neurons = 3, mask_within= True, split_by_phase = True)


# define a mask for clock and midnight neurons, respectively
clock_neurons = []
midnight_neurons = []
clock_n_mask = mouse_b_clean["cells"][:,-1]
midnight_n_mask = np.logical_and(clock_n_mask, mouse_b_clean["cells"][:,-2])

for neurons in mouse_b_clean["neurons"]:
    clock_neurons.append(neurons[clock_n_mask, :])
    midnight_neurons.append(neurons[midnight_n_mask, :])

print(f"Neuron number of recday {mouse_b['recday']} is {len(mouse_b_clean['neurons'][0])}, Midnight neuron number is {len(midnight_neurons[0])} and Clock neurons are {len(clock_neurons[0])}")    
    
b_midn_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(mouse_b_clean["rewards_configs"], mouse_b_clean["locations"], midnight_neurons, mouse_b_clean["timings"], mouse_b_clean["recday"], plotting = False, continuous = True, no_bins_per_state = 10, number_phase_neurons = 3, mask_within= True, split_by_phase = True)
b_clocks_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(mouse_b_clean["rewards_configs"], mouse_b_clean["locations"], clock_neurons, mouse_b_clean["timings"], mouse_b_clean["recday"], plotting = False, continuous = True, no_bins_per_state = 10, number_phase_neurons = 3, mask_within= True, split_by_phase = True)

if save: 
    with open(os.path.join(out_path,f"{mouse_b_clean['recday']}_res_dic"), 'wb') as f:
        pickle.dump(b_reg_result_dict, f)
    
    with open(os.path.join(out_path,f"{mouse_b_clean['recday']}_midn_dic"), 'wb') as f:
        pickle.dump(b_midn_result_dict, f)
    
    with open(os.path.join(out_path,f"{mouse_b_clean['recday']}_clock_dic"), 'wb') as f:
        pickle.dump(b_clocks_result_dict, f)



    
# mouse c
mouse_c_clean =  {}
mouse_c_clean["cells"] = mouse_c["cells"].copy()
mouse_c_clean["recday"] = mouse_c["recday"]
mouse_c_clean["rewards_configs"], mouse_c_clean["locations"], mouse_c_clean["neurons"], mouse_c_clean["timings"] = mc.simulation.analyse_ephys.clean_ephys_data(mouse_c["rewards_configs"], mouse_c["locations"], mouse_c["neurons"], mouse_c["timings"], mouse_c_clean["recday"])

#c_reg_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(mouse_c_clean["rewards_configs"], mouse_c_clean["locations"], mouse_c_clean["neurons"], mouse_c_clean["timings"], mouse_c_clean["recday"], plotting = False, continuous = True, no_bins_per_state = 10, number_phase_neurons = 3)
# whole dataset
c_reg_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(mouse_c["rewards_configs"], mouse_c["locations"], mouse_c["neurons"], mouse_c["timings"], mouse_c["recday"], plotting = False, continuous = True, no_bins_per_state = 10, number_phase_neurons = 3, mask_within= True, split_by_phase = True)


# define a mask for clock and midnight neurons, respectively
clock_neurons = []
midnight_neurons = []
clock_n_mask = mouse_c_clean["cells"][:,-1]
midnight_n_mask = np.logical_and(clock_n_mask, mouse_c_clean["cells"][:,-2])

for neurons in mouse_c_clean["neurons"]:
    clock_neurons.append(neurons[clock_n_mask, :])
    midnight_neurons.append(neurons[midnight_n_mask, :])
    
c_midn_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(mouse_c_clean["rewards_configs"], mouse_c_clean["locations"], midnight_neurons, mouse_c_clean["timings"], mouse_c_clean["recday"], plotting = False, continuous = True, no_bins_per_state = 10, number_phase_neurons = 3, mask_within= True, split_by_phase = True)
c_clocks_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(mouse_c_clean["rewards_configs"], mouse_c_clean["locations"], clock_neurons, mouse_c_clean["timings"], mouse_c_clean["recday"], plotting = False, continuous = True, no_bins_per_state = 10, number_phase_neurons = 3, mask_within= True, split_by_phase = True)

print(f"Neuron number of recday {mouse_c['recday']} is {len(mouse_c_clean['neurons'][0])}, Midnight neuron number is {len(midnight_neurons[0])} and Clock neurons are {len(clock_neurons[0])}")    

if save: 
    with open(os.path.join(out_path,f"{mouse_c_clean['recday']}_res_dic"), 'wb') as f:
        pickle.dump(c_reg_result_dict, f)
    
    with open(os.path.join(out_path,f"{mouse_c_clean['recday']}_midn_dic"), 'wb') as f:
        pickle.dump(c_midn_result_dict, f)
    
    with open(os.path.join(out_path,f"{mouse_c_clean['recday']}_clock_dic"), 'wb') as f:
        pickle.dump(c_clocks_result_dict, f)

    

# mouse d
mouse_d_clean =  {}
mouse_d_clean["cells"] = mouse_d["cells"].copy()
mouse_d_clean["recday"] = mouse_d["recday"]
mouse_d_clean["rewards_configs"], mouse_d_clean["locations"], mouse_d_clean["neurons"], mouse_d_clean["timings"] = mc.simulation.analyse_ephys.clean_ephys_data(mouse_d["rewards_configs"], mouse_d["locations"], mouse_d["neurons"], mouse_d["timings"], mouse_d_clean["recday"])

#d_reg_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(mouse_d_clean["rewards_configs"], mouse_d_clean["locations"], mouse_d_clean["neurons"], mouse_d_clean["timings"], mouse_d_clean["recday"], plotting = False, continuous = True, no_bins_per_state = 10, number_phase_neurons = 3)
# whole dataset
d_reg_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(mouse_d["rewards_configs"], mouse_d["locations"], mouse_d["neurons"], mouse_d["timings"], mouse_d["recday"], plotting = False, continuous = True, no_bins_per_state = 10, number_phase_neurons = 3, mask_within= True, split_by_phase = True)


# define a mask for clock and midnight neurons, respectively
clock_neurons = []
midnight_neurons = []
clock_n_mask = mouse_d_clean["cells"][:,-1]
midnight_n_mask = np.logical_and(clock_n_mask, mouse_d_clean["cells"][:,-2])

for neurons in mouse_d_clean["neurons"]:
    clock_neurons.append(neurons[clock_n_mask, :])
    midnight_neurons.append(neurons[midnight_n_mask, :])

print(f"Neuron number of recday {mouse_d['recday']} is {len(mouse_d_clean['neurons'][0])}, Midnight neuron number is {len(midnight_neurons[0])} and Clock neurons are {len(clock_neurons[0])}")    
    
d_midn_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(mouse_d_clean["rewards_configs"], mouse_d_clean["locations"], midnight_neurons, mouse_d_clean["timings"], mouse_d_clean["recday"], plotting = False, continuous = True, no_bins_per_state = 10, number_phase_neurons = 3, mask_within= True, split_by_phase = True)
d_clocks_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(mouse_d_clean["rewards_configs"], mouse_d_clean["locations"], clock_neurons, mouse_d_clean["timings"], mouse_d_clean["recday"], plotting = False, continuous = True, no_bins_per_state = 10, number_phase_neurons = 3, mask_within= True, split_by_phase = True)

if save: 
    with open(os.path.join(out_path,f"{mouse_d_clean['recday']}_res_dic"), 'wb') as f:
        pickle.dump(d_reg_result_dict, f)
    
    with open(os.path.join(out_path,f"{mouse_d_clean['recday']}_midn_dic"), 'wb') as f:
        pickle.dump(d_midn_result_dict, f)
    
    with open(os.path.join(out_path,f"{mouse_d_clean['recday']}_clock_dic"), 'wb') as f:
        pickle.dump(d_clocks_result_dict, f)



# mouse e
mouse_e_clean =  {}
mouse_e_clean["cells"] = mouse_e["cells"].copy()
mouse_e_clean["recday"] = mouse_e["recday"]
mouse_e_clean["rewards_configs"], mouse_e_clean["locations"], mouse_e_clean["neurons"], mouse_e_clean["timings"] = mc.simulation.analyse_ephys.clean_ephys_data(mouse_e["rewards_configs"], mouse_e["locations"], mouse_e["neurons"], mouse_e["timings"], mouse_e_clean["recday"])

#e_reg_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(mouse_e_clean["rewards_configs"], mouse_e_clean["locations"], mouse_e_clean["neurons"], mouse_e_clean["timings"], mouse_e_clean["recday"], plotting = False, continuous = True, no_bins_per_state = 10, number_phase_neurons = 3)
#whole dataset
e_reg_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(mouse_e["rewards_configs"], mouse_e["locations"], mouse_e["neurons"], mouse_e["timings"], mouse_e["recday"], plotting = False, continuous = True, no_bins_per_state = 10, number_phase_neurons = 3)


# define a mask for clock and midnight neurons, respectively
clock_neurons = []
midnight_neurons = []
clock_n_mask = mouse_e_clean["cells"][:,-1]
midnight_n_mask = np.logical_and(clock_n_mask, mouse_e_clean["cells"][:,-2])

for neurons in mouse_e_clean["neurons"]:
    clock_neurons.append(neurons[clock_n_mask, :])
    midnight_neurons.append(neurons[midnight_n_mask, :])
    
e_midn_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(mouse_e_clean["rewards_configs"], mouse_e_clean["locations"], midnight_neurons, mouse_e_clean["timings"], mouse_e_clean["recday"], plotting = False, continuous = True, no_bins_per_state = 10, number_phase_neurons = 3, mask_within= True, split_by_phase = True)
e_clocks_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(mouse_e_clean["rewards_configs"], mouse_e_clean["locations"], clock_neurons, mouse_e_clean["timings"], mouse_e_clean["recday"], plotting = False, continuous = True, no_bins_per_state = 10, number_phase_neurons = 3, mask_within= True, split_by_phase = True)

print(f"Neuron number of recday {mouse_e['recday']} is {len(mouse_e_clean['neurons'][0])}, Midnight neuron number is {len(midnight_neurons[0])} and Clock neurons are {len(clock_neurons[0])}")    

if save: 
    with open(os.path.join(out_path,f"{mouse_e_clean['recday']}_res_dic"), 'wb') as f:
        pickle.dump(e_reg_result_dict, f)
    
    with open(os.path.join(out_path,f"{mouse_e_clean['recday']}_midn_dic"), 'wb') as f:
        pickle.dump(e_midn_result_dict, f)
    
    with open(os.path.join(out_path,f"{mouse_e_clean['recday']}_clock_dic"), 'wb') as f:
        pickle.dump(e_clocks_result_dict, f)


# mouse f
mouse_f_clean =  {}
mouse_f_clean["cells"] = mouse_f["cells"].copy()
mouse_f_clean["recday"] = mouse_f["recday"]
mouse_f_clean["rewards_configs"], mouse_f_clean["locations"], mouse_f_clean["neurons"], mouse_f_clean["timings"] = mc.simulation.analyse_ephys.clean_ephys_data(mouse_f["rewards_configs"], mouse_f["locations"], mouse_f["neurons"], mouse_f["timings"], mouse_f_clean["recday"])

print('now all neurons') 
#f_reg_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(mouse_f_clean["rewards_configs"], mouse_f_clean["locations"], mouse_f_clean["neurons"], mouse_f_clean["timings"], mouse_f_clean["recday"], plotting = False, continuous = True, no_bins_per_state = 10, number_phase_neurons = 3)
# whole dataset
f_reg_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(mouse_f["rewards_configs"], mouse_f["locations"], mouse_f["neurons"], mouse_f["timings"], mouse_f["recday"], plotting = False, continuous = True, no_bins_per_state = 10, number_phase_neurons = 3, mask_within= True, split_by_phase = True)


# define a mask for clock and midnight neurons, respectively
clock_neurons = []
midnight_neurons = []
clock_n_mask = mouse_f_clean["cells"][:,-1]
midnight_n_mask = np.logical_and(clock_n_mask, mouse_f_clean["cells"][:,-2])

for neurons in mouse_f_clean["neurons"]:
    clock_neurons.append(neurons[clock_n_mask, :])
    midnight_neurons.append(neurons[midnight_n_mask, :])

print('now only midnight neurons')      
f_midn_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(mouse_f_clean["rewards_configs"], mouse_f_clean["locations"], midnight_neurons, mouse_f_clean["timings"], mouse_f_clean["recday"], plotting = False, continuous = True, no_bins_per_state = 10, number_phase_neurons = 3, mask_within= True, split_by_phase = True)
print('now only clock neurons neurons')  
f_clocks_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(mouse_f_clean["rewards_configs"], mouse_f_clean["locations"], clock_neurons, mouse_f_clean["timings"], mouse_f_clean["recday"], plotting = False, continuous = True, no_bins_per_state = 10, number_phase_neurons = 3, mask_within= True, split_by_phase = True)

print(f"Neuron number of recday {mouse_f['recday']} is {len(mouse_f_clean['neurons'][0])}, Midnight neuron number is {len(midnight_neurons[0])} and Clock neurons are {len(clock_neurons[0])}")    

if save: 
    with open(os.path.join(out_path,f"{mouse_f_clean['recday']}_res_dic"), 'wb') as f:
        pickle.dump(f_reg_result_dict, f)
    
    with open(os.path.join(out_path,f"{mouse_f_clean['recday']}_midn_dic"), 'wb') as f:
        pickle.dump(f_midn_result_dict, f)
    
    with open(os.path.join(out_path,f"{mouse_f_clean['recday']}_clock_dic"), 'wb') as f:
        pickle.dump(f_clocks_result_dict, f)


# mouse g
mouse_g_clean =  {}
mouse_g_clean["cells"] = mouse_g["cells"].copy()
mouse_g_clean["recday"] = mouse_g["recday"]
mouse_g_clean["rewards_configs"], mouse_g_clean["locations"], mouse_g_clean["neurons"], mouse_g_clean["timings"] = mc.simulation.analyse_ephys.clean_ephys_data(mouse_g["rewards_configs"], mouse_g["locations"], mouse_g["neurons"], mouse_g["timings"], mouse_g_clean["recday"])

print('now all neurons')  
#g_reg_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(mouse_g_clean["rewards_configs"], mouse_g_clean["locations"], mouse_g_clean["neurons"], mouse_g_clean["timings"], mouse_g_clean["recday"], plotting = False, continuous = True, no_bins_per_state = 10, number_phase_neurons = 3)

g_reg_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(mouse_g["rewards_configs"], mouse_g["locations"], mouse_g["neurons"], mouse_g["timings"], mouse_g["recday"], plotting = False, continuous = True, no_bins_per_state = 10, number_phase_neurons = 3, mask_within= True, split_by_phase = True)

# define a mask for clock and midnight neurons, respectively
clock_neurons = []
midnight_neurons = []
clock_n_mask = mouse_g_clean["cells"][:,-1]
midnight_n_mask = np.logical_and(clock_n_mask, mouse_g_clean["cells"][:,-2])

for neurons in mouse_g_clean["neurons"]:
    clock_neurons.append(neurons[clock_n_mask, :])
    midnight_neurons.append(neurons[midnight_n_mask, :])
    
print('now only midnight neurons')    
g_midn_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(mouse_g_clean["rewards_configs"], mouse_g_clean["locations"], midnight_neurons, mouse_g_clean["timings"], mouse_g_clean["recday"], plotting = False, continuous = True, no_bins_per_state = 10, number_phase_neurons = 3, mask_within= True, split_by_phase = True)
print('now only clock neurons neurons')  
g_clocks_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(mouse_g_clean["rewards_configs"], mouse_g_clean["locations"], clock_neurons, mouse_g_clean["timings"], mouse_g_clean["recday"], plotting = False, continuous = True, no_bins_per_state = 10, number_phase_neurons = 3, mask_within= True, split_by_phase = True)

print(f"Neuron number of recday {mouse_g['recday']} is {len(mouse_g_clean['neurons'][0])}, Midnight neuron number is {len(midnight_neurons[0])} and Clock neurons are {len(clock_neurons[0])}")    

if save: 
    with open(os.path.join(out_path,f"{mouse_g_clean['recday']}_res_dic"), 'wb') as f:
        pickle.dump(g_reg_result_dict, f)
    
    with open(os.path.join(out_path,f"{mouse_g_clean['recday']}_midn_dic"), 'wb') as f:
        pickle.dump(g_midn_result_dict, f)
    
    with open(os.path.join(out_path,f"{mouse_g_clean['recday']}_clock_dic"), 'wb') as f:
        pickle.dump(g_clocks_result_dict, f)


# mouse h
mouse_h_clean =  {}
mouse_h_clean["cells"] = mouse_h["cells"].copy()
mouse_h_clean["recday"] = mouse_h["recday"]
mouse_h_clean["rewards_configs"], mouse_h_clean["locations"], mouse_h_clean["neurons"], mouse_h_clean["timings"] = mc.simulation.analyse_ephys.clean_ephys_data(mouse_h["rewards_configs"], mouse_h["locations"], mouse_h["neurons"], mouse_h["timings"], mouse_h_clean["recday"])

#h_reg_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(mouse_h_clean["rewards_configs"], mouse_h_clean["locations"], mouse_h_clean["neurons"], mouse_h_clean["timings"], mouse_h_clean["recday"], plotting = True, continuous = True, no_bins_per_state = 10, number_phase_neurons = 3)

# compare what happens if I take the whole dataset.
h_reg_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(mouse_h["rewards_configs"], mouse_h["locations"], mouse_h["neurons"], mouse_h["timings"], mouse_h["recday"], plotting = True, continuous = True, no_bins_per_state = 10, number_phase_neurons = 3, mask_within= True, split_by_phase = True)



# define a mask for clock and midnight neurons, respectively
clock_neurons = []
midnight_neurons = []
clock_n_mask = mouse_h_clean["cells"][:,-1]
midnight_n_mask = np.logical_and(clock_n_mask, mouse_h_clean["cells"][:,-2])

for neurons in mouse_h_clean["neurons"]:
    clock_neurons.append(neurons[clock_n_mask, :])
    midnight_neurons.append(neurons[midnight_n_mask, :])
    
h_midn_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(mouse_h_clean["rewards_configs"], mouse_h_clean["locations"], midnight_neurons, mouse_h_clean["timings"], mouse_h_clean["recday"], plotting = False, continuous = True, no_bins_per_state = 10, number_phase_neurons = 3, mask_within= True, split_by_phase = True)
h_clocks_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(mouse_h_clean["rewards_configs"], mouse_h_clean["locations"], clock_neurons, mouse_h_clean["timings"], mouse_h_clean["recday"], plotting = False, continuous = True, no_bins_per_state = 10, number_phase_neurons = 3, mask_within= True, split_by_phase = True)

print(f"Neuron number of recday {mouse_h['recday']} is {len(mouse_h_clean['neurons'][0])}, Midnight neuron number is {len(midnight_neurons[0])} and Clock neurons are {len(clock_neurons[0])}")    

if save: 
    with open(os.path.join(out_path,f"{mouse_h_clean['recday']}_res_dic"), 'wb') as f:
        pickle.dump(h_reg_result_dict, f)
    
    with open(os.path.join(out_path,f"{mouse_h_clean['recday']}_midn_dic"), 'wb') as f:
        pickle.dump(h_midn_result_dict, f)
    
    with open(os.path.join(out_path,f"{mouse_h_clean['recday']}_clock_dic"), 'wb') as f:
        pickle.dump(h_clocks_result_dict, f)


# THIS CAN BE EASILY ADJUSTED, JUST COMMENTED OUT SO I CAN STORE THE RESULTS!!



# # PLOT THE RESULTS OF THIS!!
# # one plot: all neurons
# data_all = [[a_reg_result_dict["reg_early_phase_midnight-clocks"][0],b_reg_result_dict["reg_early_phase_midnight-clocks"][0], c_reg_result_dict["reg_early_phase_midnight-clocks"][0], d_reg_result_dict["reg_early_phase_midnight-clocks"][0], e_reg_result_dict["reg_early_phase_midnight-clocks"][0]],
#               [a_reg_result_dict["reg_early_phase_midnight-clocks"][1],b_reg_result_dict["reg_early_phase_midnight-clocks"][1], c_reg_result_dict["reg_early_phase_midnight-clocks"][1], d_reg_result_dict["reg_early_phase_midnight-clocks"][1], e_reg_result_dict["reg_early_phase_midnight-clocks"][1]],
#               [a_reg_result_dict["reg_mid_phase_midnight-clocks"][0],b_reg_result_dict["reg_mid_phase_midnight-clocks"][0], c_reg_result_dict["reg_mid_phase_midnight-clocks"][0], d_reg_result_dict["reg_mid_phase_midnight-clocks"][0], e_reg_result_dict["reg_mid_phase_midnight-clocks"][0]],
#               [a_reg_result_dict["reg_mid_phase_midnight-clocks"][1],b_reg_result_dict["reg_mid_phase_midnight-clocks"][1], c_reg_result_dict["reg_mid_phase_midnight-clocks"][1], d_reg_result_dict["reg_mid_phase_midnight-clocks"][1], e_reg_result_dict["reg_mid_phase_midnight-clocks"][1]],
#               [a_reg_result_dict["reg_late_phase_midnight-clocks"][0],b_reg_result_dict["reg_late_phase_midnight-clocks"][0], c_reg_result_dict["reg_late_phase_midnight-clocks"][0], d_reg_result_dict["reg_late_phase_midnight-clocks"][0], e_reg_result_dict["reg_late_phase_midnight-clocks"][0]],
#               [a_reg_result_dict["reg_late_phase_midnight-clocks"][1],b_reg_result_dict["reg_late_phase_midnight-clocks"][1], c_reg_result_dict["reg_late_phase_midnight-clocks"][1], d_reg_result_dict["reg_late_phase_midnight-clocks"][1], e_reg_result_dict["reg_late_phase_midnight-clocks"][1]],
#               [a_reg_result_dict["reg_all_reversedphase_midnight-clocks"][0],b_reg_result_dict["reg_all_reversedphase_midnight-clocks"][0], c_reg_result_dict["reg_all_reversedphase_midnight-clocks"][0], d_reg_result_dict["reg_all_reversedphase_midnight-clocks"][0], e_reg_result_dict["reg_all_reversedphase_midnight-clocks"][0]],
#               [a_reg_result_dict["reg_all_reversedphase_midnight-clocks"][1],b_reg_result_dict["reg_all_reversedphase_midnight-clocks"][1], c_reg_result_dict["reg_all_reversedphase_midnight-clocks"][1], d_reg_result_dict["reg_all_reversedphase_midnight-clocks"][1], e_reg_result_dict["reg_all_reversedphase_midnight-clocks"][1]]]


# label_string_list_all = ['early midight', 'early clocks', 'mid midnight', 'mid clocks', 'late midnight', 'late clocks', 'put together midnight', 'put together clocks']
# label_tick_list_all = [0,1,2,3,4,5,6,7]
# title_string_all = 'All neurons included, averaged over trials, between tasks, binned in 10 bins per state, continuous model'

# mc.simulation.analyse_ephys.plotting_hist_scat(data_all, label_string_list_all, label_tick_list_all, title_string_all)


# # second plot: only midnight
# data_midn = [[a_midn_result_dict["reg_early_phase_midnight-clocks"][0],b_midn_result_dict["reg_early_phase_midnight-clocks"][0], c_midn_result_dict["reg_early_phase_midnight-clocks"][0], d_midn_result_dict["reg_early_phase_midnight-clocks"][0], e_midn_result_dict["reg_early_phase_midnight-clocks"][0]],
#               [a_midn_result_dict["reg_early_phase_midnight-clocks"][1],b_midn_result_dict["reg_early_phase_midnight-clocks"][1], c_midn_result_dict["reg_early_phase_midnight-clocks"][1], d_midn_result_dict["reg_early_phase_midnight-clocks"][1], e_midn_result_dict["reg_early_phase_midnight-clocks"][1]],
#               [a_midn_result_dict["reg_mid_phase_midnight-clocks"][0],b_midn_result_dict["reg_mid_phase_midnight-clocks"][0], c_midn_result_dict["reg_mid_phase_midnight-clocks"][0], d_midn_result_dict["reg_mid_phase_midnight-clocks"][0], e_midn_result_dict["reg_mid_phase_midnight-clocks"][0]],
#               [a_midn_result_dict["reg_mid_phase_midnight-clocks"][1],b_midn_result_dict["reg_mid_phase_midnight-clocks"][1], c_midn_result_dict["reg_mid_phase_midnight-clocks"][1], d_midn_result_dict["reg_mid_phase_midnight-clocks"][1], e_midn_result_dict["reg_mid_phase_midnight-clocks"][1]],
#               [a_midn_result_dict["reg_late_phase_midnight-clocks"][0],b_midn_result_dict["reg_late_phase_midnight-clocks"][0], c_midn_result_dict["reg_late_phase_midnight-clocks"][0], d_midn_result_dict["reg_late_phase_midnight-clocks"][0], e_midn_result_dict["reg_late_phase_midnight-clocks"][0]],
#               [a_midn_result_dict["reg_late_phase_midnight-clocks"][1],b_midn_result_dict["reg_late_phase_midnight-clocks"][1], c_midn_result_dict["reg_late_phase_midnight-clocks"][1], d_midn_result_dict["reg_late_phase_midnight-clocks"][1], e_midn_result_dict["reg_late_phase_midnight-clocks"][1]],
#               [a_midn_result_dict["reg_all_reversedphase_midnight-clocks"][0],b_midn_result_dict["reg_all_reversedphase_midnight-clocks"][0], c_midn_result_dict["reg_all_reversedphase_midnight-clocks"][0], d_midn_result_dict["reg_all_reversedphase_midnight-clocks"][0], e_midn_result_dict["reg_all_reversedphase_midnight-clocks"][0]],
#               [a_midn_result_dict["reg_all_reversedphase_midnight-clocks"][1],b_midn_result_dict["reg_all_reversedphase_midnight-clocks"][1], c_midn_result_dict["reg_all_reversedphase_midnight-clocks"][1], d_midn_result_dict["reg_all_reversedphase_midnight-clocks"][1], e_midn_result_dict["reg_all_reversedphase_midnight-clocks"][1]]]


# label_string_list_all = ['early midight', 'early clocks', 'mid midnight', 'mid clocks', 'late midnight', 'late clocks', 'put together midnight', 'put together clocks']
# label_tick_list_all = [0,1,2,3,4,5,6,7]
# title_string_all = 'Only MIDNIGHT neurons included, averaged over trials, between tasks, binned in 10 bins per state, continuous model'

# mc.simulation.analyse_ephys.plotting_hist_scat(data_midn, label_string_list_all, label_tick_list_all, title_string_all)


# # third plot: the whole clocks
# data_clocks = [[a_clocks_result_dict["reg_early_phase_midnight-clocks"][0],b_clocks_result_dict["reg_early_phase_midnight-clocks"][0], c_clocks_result_dict["reg_early_phase_midnight-clocks"][0], d_clocks_result_dict["reg_early_phase_midnight-clocks"][0], e_clocks_result_dict["reg_early_phase_midnight-clocks"][0]],
#               [a_clocks_result_dict["reg_early_phase_midnight-clocks"][1],b_clocks_result_dict["reg_early_phase_midnight-clocks"][1], c_clocks_result_dict["reg_early_phase_midnight-clocks"][1], d_clocks_result_dict["reg_early_phase_midnight-clocks"][1], e_clocks_result_dict["reg_early_phase_midnight-clocks"][1]],
#               [a_clocks_result_dict["reg_mid_phase_midnight-clocks"][0],b_clocks_result_dict["reg_mid_phase_midnight-clocks"][0], c_clocks_result_dict["reg_mid_phase_midnight-clocks"][0], d_clocks_result_dict["reg_mid_phase_midnight-clocks"][0], e_clocks_result_dict["reg_mid_phase_midnight-clocks"][0]],
#               [a_clocks_result_dict["reg_mid_phase_midnight-clocks"][1],b_clocks_result_dict["reg_mid_phase_midnight-clocks"][1], c_clocks_result_dict["reg_mid_phase_midnight-clocks"][1], d_clocks_result_dict["reg_mid_phase_midnight-clocks"][1], e_clocks_result_dict["reg_mid_phase_midnight-clocks"][1]],
#               [a_clocks_result_dict["reg_late_phase_midnight-clocks"][0],b_clocks_result_dict["reg_late_phase_midnight-clocks"][0], c_clocks_result_dict["reg_late_phase_midnight-clocks"][0], d_clocks_result_dict["reg_late_phase_midnight-clocks"][0], e_clocks_result_dict["reg_late_phase_midnight-clocks"][0]],
#               [a_clocks_result_dict["reg_late_phase_midnight-clocks"][1],b_clocks_result_dict["reg_late_phase_midnight-clocks"][1], c_clocks_result_dict["reg_late_phase_midnight-clocks"][1], d_clocks_result_dict["reg_late_phase_midnight-clocks"][1], e_clocks_result_dict["reg_late_phase_midnight-clocks"][1]],
#               [a_clocks_result_dict["reg_all_reversedphase_midnight-clocks"][0],b_clocks_result_dict["reg_all_reversedphase_midnight-clocks"][0], c_clocks_result_dict["reg_all_reversedphase_midnight-clocks"][0], d_clocks_result_dict["reg_all_reversedphase_midnight-clocks"][0], e_clocks_result_dict["reg_all_reversedphase_midnight-clocks"][0]],
#               [a_clocks_result_dict["reg_all_reversedphase_midnight-clocks"][1],b_clocks_result_dict["reg_all_reversedphase_midnight-clocks"][1], c_clocks_result_dict["reg_all_reversedphase_midnight-clocks"][1], d_clocks_result_dict["reg_all_reversedphase_midnight-clocks"][1], e_clocks_result_dict["reg_all_reversedphase_midnight-clocks"][1]]]


# label_string_list_all = ['early midight', 'early clocks', 'mid midnight', 'mid clocks', 'late midnight', 'late clocks', 'put together midnight', 'put together clocks']
# label_tick_list_all = [0,1,2,3,4,5,6,7]
# title_string_all = 'Only CLOCK neurons included, averaged over trials, between tasks, binned in 10 bins per state, continuous model'

# mc.simulation.analyse_ephys.plotting_hist_scat(data_clocks, label_string_list_all, label_tick_list_all, title_string_all)


# # build another figure.
# # this time, build it with the normal regression
# letters = ['a','b','c','d','e','f', 'g', 'h']
# list_allneurons =[]
# list_midn = []
# list_clockn = []
# for i, mouse in enumerate(letters):
#     dataset_all = eval(f"{mouse}_reg_result_dict")
#     dataset_midn = eval(f"{mouse}_midn_result_dict")
#     dataset_clock = eval(f"{mouse}_clocks_result_dict")
#     list_allneurons.append(dataset_all['reg_all_midnight-clocks-loc-phase'])
#     list_midn.append(dataset_midn['reg_all_midnight-clocks-loc-phase'])
#     list_clockn.append(dataset_clock['reg_all_midnight-clocks-loc-phase'])

# plotting_clockn = np.empty((len(list_clockn[0]), len(list_clockn)))
# for col, regressor in enumerate(range(0, len(list_clockn[0]))):
#     for row, dataset in enumerate(range(0, len(list_clockn))):
#         plotting_clockn[col, row]= list_clockn[dataset][regressor]

# list_plot_clock = []
# for row in plotting_clockn:
#     list_plot_clock.append(row)

# plotting_midn = np.empty((len(list_midn[0]), len(list_midn)))
# for col, regressor in enumerate(range(0, len(list_midn[0]))):
#     for row, dataset in enumerate(range(0, len(list_midn))):
#         plotting_midn[col, row]= list_midn[dataset][regressor]

# list_plot_midn = []
# for row in plotting_midn:
#     list_plot_midn.append(row)

# plotting_all = np.empty((len(list_allneurons[0]), len(list_allneurons)))
# for col, regressor in enumerate(range(0, len(list_allneurons[0]))):
#     for row, dataset in enumerate(range(0, len(list_allneurons))):
#         plotting_all[col, row]= list_allneurons[dataset][regressor]

# list_plot_all = []
# for row in plotting_all:
#     list_plot_all.append(row)
     
# for elem in list_plot_midn:
#     list_plot_all.append(elem)

# for elem in list_plot_midn:
#     list_plot_all.append(elem)

# labels_compl_reg = ['all_midn', 'all_clocks', 'all_loc', 'all_phase', 'clock_midn', 'clock_clocks', 'clock_loc', 'clock_phase','midn_midn', 'midn_clocks', 'midn_loc', 'midn_phase']
# label_ticks_compl_reg = [1,2,3,4,5,6,7,8,9,10,11,12]
# title_compl_reg = ['Complete regression, leaving neurons out'] 
# mc.simulation.analyse_ephys.plotting_hist_scat(list_plot_all, labels_compl_reg, label_ticks_compl_reg, title_compl_reg)   
 
 

# # build another figure.
# # use the early-mid-late weights
# letters = ['a','b','c','d','e','f', 'g', 'h']
# list_allneurons =[]
# list_midn = []
# list_clockn = []
# for part in ['reg_early_phase_midnight-clocks', 'reg_mid_phase_midnight-clocks', 'reg_late_phase_midnight-clocks']:
#     for i, mouse in enumerate(letters):
#         dataset_all = eval(f"{mouse}_reg_result_dict")
#         dataset_midn = eval(f"{mouse}_midn_result_dict")
#         dataset_clock = eval(f"{mouse}_clocks_result_dict")    
#         list_allneurons.append(dataset_all[part])
#         list_midn.append(dataset_midn[part])
#         list_clockn.append(dataset_clock[part])

# early_midnight_all = []
# early_clock_all = []
# early_midnight_midn = []
# early_clock_midn = []
# early_midnight_clo = []
# early_clock_clo = []
# for dataset,x in enumerate(letters):
#     early_midnight_all.append(list_allneurons[dataset][0])
#     early_clock_all.append(list_allneurons[dataset][1])
#     early_midnight_midn.append(list_midn[dataset][0])
#     early_clock_midn.append(list_midn[dataset][1])
#     early_midnight_clo.append(list_clockn[dataset][0])
#     early_clock_clo.append(list_clockn[dataset][1])

# mid_midnight_all = []
# mid_clocks_all = []
# mid_midnight_midn = []
# mid_clock_midn = []
# mid_midnight_clo = []
# mid_clock_clo = []
# for dataset,x in enumerate(letters):
#     mid_midnight_all.append(list_allneurons[dataset+len(letters)][0])
#     mid_clocks_all.append(list_allneurons[dataset+len(letters)][1])
#     mid_midnight_midn.append(list_midn[dataset+len(letters)][0])
#     mid_clock_midn.append(list_midn[dataset+len(letters)][1])
#     mid_midnight_clo.append(list_clockn[dataset+len(letters)][0])
#     mid_clock_clo.append(list_clockn[dataset+len(letters)][1])

# late_midnight_all = []
# late_clocks_all = []
# late_midnight_midn = []
# late_clock_midn = []
# late_midnight_clo = []
# late_clock_clo = []
# for dataset,x in enumerate(letters):
#     late_midnight_all.append(list_allneurons[dataset+len(letters)*2][0])
#     late_clocks_all.append(list_allneurons[dataset+len(letters)*2][1])
#     late_midnight_midn.append(list_midn[dataset+len(letters)*2][0])
#     late_clock_midn.append(list_midn[dataset+len(letters)*2][1])
#     late_midnight_clo.append(list_clockn[dataset+len(letters)*2][0])
#     late_clock_clo.append(list_clockn[dataset+len(letters)*2][1])


# plotting_early = [early_midnight_all, early_midnight_clo, early_midnight_midn, early_clock_all, early_clock_clo, early_clock_midn]    

# labels_early_reg = ['midnight_all', 'midnight_clo', 'midnight_midn', 'clock_all', 'clock_clo', 'clock_midn']
# label_ticks_early_reg = [1,2,3,4,5,6]
# title_early_reg = ['Early comparing all, only clock and only midn neurons'] 
# mc.simulation.analyse_ephys.plotting_hist_scat(plotting_early, labels_early_reg, label_ticks_early_reg, title_early_reg)   
 
# plotting_mid = [mid_midnight_all, mid_midnight_clo, mid_midnight_midn, mid_clocks_all, mid_clock_clo, mid_clock_midn]    

# labels_mid_reg = ['midnight_all', 'midnight_clo', 'midnight_midn', 'clock_all', 'clock_clo', 'clock_midn']
# label_ticks_mid_reg = [1,2,3,4,5,6]
# title_mid_reg = ['Mid comparing all, only clock and only midn neurons'] 
# mc.simulation.analyse_ephys.plotting_hist_scat(plotting_mid, labels_mid_reg, label_ticks_mid_reg, title_mid_reg)   

# plotting_late = [late_midnight_all, late_midnight_clo, late_midnight_midn, late_clocks_all, late_clock_clo, late_clock_midn]    

# labels_late_reg = ['midnight_all', 'midnight_clo', 'midnight_midn', 'clock_all', 'clock_clo', 'clock_midn']
# label_ticks_late_reg = [1,2,3,4,5,6]
# title_late_reg = ['Late comparing all, only clock and only midn neurons'] 
# mc.simulation.analyse_ephys.plotting_hist_scat(plotting_late, labels_late_reg, label_ticks_late_reg, title_late_reg)   


################################################################################

 
    
# # PART 2: 


# regression_mouse_a,  contrasts_mouse_a = mc.simulation.analyse_ephys.reg_per_task_config(a_rewards_configs, a_locations, a_neurons, a_timings, contrast_matrix, continuous = True, no_bins_per_state = 3, mouse_recday = 'me11_05122021_06122021')
# regression_mouse_b,  contrasts_mouse_b = mc.simulation.analyse_ephys.reg_per_task_config(b_rewards_configs, b_locations, b_neurons, b_timings, contrast_matrix, continuous = True, no_bins_per_state = 3, mouse_recday = 'me11_01122021_02122021')

# regression_mouse_a_11,  contrasts_mouse_a_11 = mc.simulation.analyse_ephys.reg_per_task_config(a_rewards_configs, a_locations, a_neurons, a_timings, contrast_matrix, continuous = True, no_bins_per_state = 11, mouse_recday = 'me11_05122021_06122021')
# regression_mouse_b_11,  contrasts_mouse_b_11 = mc.simulation.analyse_ephys.reg_per_task_config(b_rewards_configs, b_locations, b_neurons, b_timings, contrast_matrix, continuous = True, no_bins_per_state = 11, mouse_recday = 'me11_01122021_02122021')


# a_reg_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(a_rewards_configs, a_locations, a_neurons, a_timings, mouse_recday = 'me11_05122021_06122021', plotting = True, continuous = True)

# a_reg_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(a_rewards_configs, a_locations, a_neurons, a_timings, mouse_recday = 'me11_05122021_06122021', plotting = False, continuous = True)
# b_reg_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(b_rewards_configs, b_locations, b_neurons, b_timings, mouse_recday = 'me11_01122021_02122021', plotting = True, continuous = True)
# c_reg_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(c_rewards_configs, c_locations, c_neurons, c_timings, mouse_recday = 'me10_09122021_10122021', plotting = False, continuous = True)
# d_reg_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(d_rewards_configs, d_locations, d_neurons, d_timings, mouse_recday = 'me08_10092021_11092021', plotting = False, continuous = True)
# e_reg_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(e_rewards_configs, e_locations, e_neurons, e_timings, mouse_recday = 'ah04_09122021_10122021', plotting = False, continuous = True)
# f_reg_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(f_rewards_configs, f_locations, f_neurons, f_timings, mouse_recday = 'ah04_05122021_06122021', plotting = False, continuous = True)
# g_reg_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(g_rewards_configs, g_locations, g_neurons, g_timings, mouse_recday = 'ah04_01122021_02122021', plotting = False, continuous = True)
# h_reg_result_dict = mc.simulation.analyse_ephys.reg_across_tasks(h_rewards_configs, h_locations, h_neurons, h_timings, mouse_recday = 'ah03_18082021_19082021', plotting = False, continuous = True)


# # results_reg_acro_mouse_b, scipy_reg_acro_mouse_b, coefficients_acro_mouse_b, averaged_reg_results = mc.simulation.single_sub_ephys.reg_across_tasks(b_rewards_configs, b_locations, b_neurons, b_timings, mouse_recday = 'me11_01122021_02122021')

# # results_reg_acro_mouse_c, scipy_reg_acro_mouse_c, coefficients_acro_mouse_c, averaged_reg_results = mc.simulation.single_sub_ephys.reg_across_tasks(c_rewards_configs, c_locations, c_neurons, c_timings, mouse_recday = 'me10_09122021_10122021')
# # results_reg_acro_mouse_d, scipy_reg_acro_mouse_d, coefficients_acro_mouse_d, averaged_reg_results = mc.simulation.single_sub_ephys.reg_across_tasks(d_rewards_configs, d_locations, d_neurons, d_timings, mouse_recday = 'me08_10092021_11092021')

# # results_reg_acro_mouse_e, scipy_reg_acro_mouse_e, coefficients_acro_mouse_e, averaged_reg_results = mc.simulation.single_sub_ephys.reg_across_tasks(e_rewards_configs, e_locations, e_neurons, e_timings, mouse_recday = 'ah04_09122021_10122021')
# # results_reg_acro_mouse_f, scipy_reg_acro_mouse_f, coefficients_acro_mouse_f, averaged_reg_results = mc.simulation.single_sub_ephys.reg_across_tasks(f_rewards_configs, f_locations, f_neurons, f_timings, mouse_recday = 'ah04_05122021_06122021')

# #results_reg_acro_mouse_g, scipy_reg_acro_mouse_g, coefficients_acro_mouse_g, averaged_reg_results = mc.simulation.single_sub_ephys.reg_across_tasks(g_rewards_configs, g_locations, g_neurons, g_timings, mouse_recday = 'ah04_01122021_02122021')
# # results_reg_acro_mouse_h, scipy_reg_acro_mouse_h, coefficients_acro_mouse_h, averaged_reg_results = mc.simulation.single_sub_ephys.reg_across_tasks(h_rewards_configs, h_locations, h_neurons, h_timings, mouse_recday = 'ah03_18082021_19082021')


# # # THIS IS THE INTERESTIGN THING RN
# # import pdb; pdb.set_trace()



# # now generate the average beta value for each model
# mean_beta_clocks_a = list()
# mean_beta_midnight_a = list()
# mean_beta_locations_a = list()
# mean_beta_phase_a = list()
# mean_contrasts_mouse_a = np.zeros((len(contrast_matrix), len(regression_mouse_a)))
# for task_no, betas in enumerate(regression_mouse_a):
#     mean_beta_clocks_a.append(np.mean(betas[:,0]))
#     mean_beta_midnight_a.append(np.mean(betas[:,1]))
#     mean_beta_locations_a.append(np.mean(betas[:,2]))
#     mean_beta_phase_a.append(np.mean(betas[:,3]))
#     for contr in range(len(contrast_matrix)):
#         mean_contrasts_mouse_a[contr, task_no] = np.mean(contrasts_mouse_a[task_no][contr])

# mean_beta_clocks_b = list()
# mean_beta_midnight_b = list()
# mean_beta_locations_b = list()
# mean_beta_phase_b = list()
# mean_contrasts_mouse_b = np.zeros((len(contrast_matrix), len(regression_mouse_b)))
# for task_no, betas in enumerate(regression_mouse_b):
#     mean_beta_clocks_b.append(np.mean(betas[:,0]))
#     mean_beta_midnight_b.append(np.mean(betas[:,1]))
#     mean_beta_locations_b.append(np.mean(betas[:,2]))
#     mean_beta_phase_b.append(np.mean(betas[:,3]))
#     for contr in range(len(contrast_matrix)):
#         mean_contrasts_mouse_b[contr, task_no] = np.mean(contrasts_mouse_b[task_no][contr])


# mean_contrasts_mouse_a_11 = np.zeros((len(contrast_matrix), len(regression_mouse_a_11)))
# for task_no, betas in enumerate(regression_mouse_a_11):
#     for contr in range(len(contrast_matrix)):
#         mean_contrasts_mouse_a_11[contr, task_no] = np.mean(contrasts_mouse_a_11[task_no][contr])


# mean_contrasts_mouse_b_11 = np.zeros((len(contrast_matrix), len(regression_mouse_b_11)))
# for task_no, betas in enumerate(regression_mouse_b_11):
#     for contr in range(len(contrast_matrix)):
#         mean_contrasts_mouse_b_11[contr, task_no] = np.mean(contrasts_mouse_b_11[task_no][contr])


    
# # regression_mouse_b,  contrasts_mouse_b = mc.simulation.single_sub_ephys.reg_per_task_config(b_rewards_configs, b_locations, b_neurons, b_timings, contrast_matrix)
# # # now generate the average beta value for each model
# # mean_beta_clocks_b = list()
# # mean_beta_midnight_b = list()
# # mean_beta_locations_b = list()
# # mean_beta_phase_b = list()
# # mean_contrasts_mouse_b = np.zeros((len(contrast_matrix), len(regression_mouse_b)))
# # for task_no, betas in enumerate(regression_mouse_b):
# #     mean_beta_clocks_b.append(np.mean(betas[:,0]))
# #     mean_beta_midnight_b.append(np.mean(betas[:,1]))
# #     mean_beta_locations_b.append(np.mean(betas[:,2]))
# #     mean_beta_phase_b.append(np.mean(betas[:,3]))
# #     for contr in range(len(contrast_matrix)):
# #         mean_contrasts_mouse_b[contr, task_no] = np.mean(contrasts_mouse_b[task_no][contr])

# # # # SOMETHING GOES WRONG HERE, SUPER LARGE betas
# # # regression_mouse_c,  contrasts_mouse_c = mc.simulation.single_sub_ephys.reg_per_task_config(c_rewards_configs, c_locations, c_neurons, c_timings, contrast_matrix)
# # # # for some reason, for the 5th run (timepoints:[741, 776, 794, 811, 818])
# # # # the mouse just stays at one location (1)
# # # # now generate the average beta value for each model
# # # mean_beta_clocks_c = list()
# # # mean_beta_midnight_c = list()
# # # mean_beta_locations_c = list()
# # # mean_contrasts_mouse_c = np.zeros((len(contrast_matrix), len(regression_mouse_c)))
# # # for task_no, betas in enumerate(regression_mouse_c):
# # #     mean_beta_clocks_c.append(np.mean(betas[:,0]))
# # #     mean_beta_midnight_c.append(np.mean(betas[:,1]))
# # #     mean_beta_locations_c.append(np.mean(betas[:,2]))
# # #     for contr in range(len(contrast_matrix)):
# # #         mean_contrasts_mouse_c[contr, task_no] = np.mean(contrasts_mouse_c[task_no][contr])

# # # # SOMETHING GOES WRONG HERE, SUPER SMALLbetas    
# # # regression_mouse_d,  contrasts_mouse_d = mc.simulation.single_sub_ephys.reg_per_task_config(d_rewards_configs[0:6,:], d_locations, d_neurons, d_timings, contrast_matrix)
# # # # now generate the average beta value for each model
# # # mean_beta_clocks_d = list()
# # # mean_beta_midnight_d = list()
# # # mean_beta_locations_d = list()
# # # mean_contrasts_mouse_d = np.zeros((len(contrast_matrix), len(regression_mouse_d)))
# # # for task_no, betas in enumerate(regression_mouse_d):
# # #     mean_beta_clocks_d.append(np.mean(betas[:,0]))
# # #     mean_beta_midnight_d.append(np.mean(betas[:,1]))
# # #     mean_beta_locations_d.append(np.mean(betas[:,2]))
# # #     for contr in range(len(contrast_matrix)):
# # #         mean_contrasts_mouse_d[contr, task_no] = np.mean(contrasts_mouse_d[task_no][contr])

# # # # SOMETHING GOES WRONG HERE, SUPER SMALLbetas    
# # # regression_mouse_e,  contrasts_mouse_e = mc.simulation.single_sub_ephys.reg_per_task_config(e_rewards_configs, e_locations, e_neurons, e_timings, contrast_matrix)
# # # # now generate the average beta value for each model
# # # mean_beta_clocks_e = list()
# # # mean_beta_midnight_e = list()
# # # mean_beta_locations_e = list()
# # # mean_contrasts_mouse_e = np.zeros((len(contrast_matrix), len(regression_mouse_e)))
# # # for task_no, betas in enumerate(regression_mouse_e):
# # #     mean_beta_clocks_e.append(np.mean(betas[:,0]))
# # #     mean_beta_midnight_e.append(np.mean(betas[:,1]))
# # #     mean_beta_locations_e.append(np.mean(betas[:,2]))
# # #     for contr in range(len(contrast_matrix)):
# # #         mean_contrasts_mouse_e[contr, task_no] = np.mean(contrasts_mouse_e[task_no][contr])
    
# # # regression_mouse_f,  contrasts_mouse_f = mc.simulation.single_sub_ephys.reg_per_task_config(f_rewards_configs, f_locations, f_neurons, f_timings, contrast_matrix)
# # # # now generate the average beta value for each model
# # # mean_beta_clocks_f = list()
# # # mean_beta_midnight_f = list()
# # # mean_beta_locations_f = list()
# # # mean_contrasts_mouse_f = np.zeros((len(contrast_matrix), len(regression_mouse_f)))
# # # for task_no, betas in enumerate(regression_mouse_f):
# # #     mean_beta_clocks_f.append(np.mean(betas[:,0]))
# # #     mean_beta_midnight_f.append(np.mean(betas[:,1]))
# # #     mean_beta_locations_f.append(np.mean(betas[:,2]))
# # #     for contr in range(len(contrast_matrix)):
# # #         mean_contrasts_mouse_f[contr, task_no] = np.mean(contrasts_mouse_f[task_no][contr])
    
# # regression_mouse_g,  contrasts_mouse_g = mc.simulation.single_sub_ephys.reg_per_task_config(g_rewards_configs, g_locations, g_neurons, g_timings, contrast_matrix)
# # # now generate the average beta value for each model
# # mean_beta_clocks_g = list()
# # mean_beta_midnight_g = list()
# # mean_beta_locations_g = list()
# # mean_beta_phase_g = list()
# # mean_contrasts_mouse_g = np.zeros((len(contrast_matrix), len(regression_mouse_g)))
# # for task_no, betas in enumerate(regression_mouse_g):
# #     mean_beta_clocks_g.append(np.mean(betas[:,0]))
# #     mean_beta_midnight_g.append(np.mean(betas[:,1]))
# #     mean_beta_locations_g.append(np.mean(betas[:,2]))   
# #     mean_beta_phase_g.append(np.mean(betas[:,3]))
# #     for contr in range(len(contrast_matrix)):
# #         mean_contrasts_mouse_g[contr, task_no] = np.mean(contrasts_mouse_g[task_no][contr])
 


# # # regression_mouse_h, contrasts_mouse_h = mc.simulation.single_sub_ephys.reg_per_task_config(h_rewards_configs, h_locations, h_neurons, h_timings, contrast_matrix)
# # # # now generate the average beta value for each model
# # # mean_beta_clocks_h = list()
# # # mean_beta_midnight_h = list()
# # # mean_beta_locations_h = list()
# # # mean_contrasts_mouse_h = np.zeros((len(contrast_matrix), len(regression_mouse_h)))
# # # for task_no, betas in enumerate(regression_mouse_h):
# # #     mean_beta_clocks_h.append(np.mean(betas[:,0]))
# # #     mean_beta_midnight_h.append(np.mean(betas[:,1]))
# # #     mean_beta_locations_h.append(np.mean(betas[:,2]))
# # #     for contr in range(len(contrast_matrix)):
# # #         mean_contrasts_mouse_h[contr, task_no] = np.mean(contrasts_mouse_h[task_no][contr])


# # #
# # # Part 3: Plotting

# # interim plotting. 


# # also plot one violin per regressor across subjects, for the different models
# # early
# data_early_mid = [a_reg_result_dict["reg_early_phase_midnight-clocks"][0],b_reg_result_dict["reg_early_phase_midnight-clocks"][0], c_reg_result_dict["reg_early_phase_midnight-clocks"][0], d_reg_result_dict["reg_early_phase_midnight-clocks"][0], e_reg_result_dict["reg_early_phase_midnight-clocks"][0], f_reg_result_dict["reg_early_phase_midnight-clocks"][0], g_reg_result_dict["reg_early_phase_midnight-clocks"][0], h_reg_result_dict["reg_early_phase_midnight-clocks"][0]]
# data_early_clock = [a_reg_result_dict["reg_early_phase_midnight-clocks"][1],b_reg_result_dict["reg_early_phase_midnight-clocks"][1], c_reg_result_dict["reg_early_phase_midnight-clocks"][1], d_reg_result_dict["reg_early_phase_midnight-clocks"][1], e_reg_result_dict["reg_early_phase_midnight-clocks"][1], f_reg_result_dict["reg_early_phase_midnight-clocks"][1], g_reg_result_dict["reg_early_phase_midnight-clocks"][1], h_reg_result_dict["reg_early_phase_midnight-clocks"][1]]

# hist_data_early = [data_early_mid, data_early_clock]
# fig_early, ax_early = plt.subplots()
# ax_early.violinplot(hist_data_early, showmedians = True)
# ax_early.set_title('Betas only early phase, z-scored data')
# ax_early.set_xticks([1,2])
# ax_early.set_xticklabels(["Midnight", "Clocks"])


# # middle 
# data_mid_mid = [a_reg_result_dict["reg_mid_phase_midnight-clocks"][0],b_reg_result_dict["reg_mid_phase_midnight-clocks"][0], c_reg_result_dict["reg_mid_phase_midnight-clocks"][0], d_reg_result_dict["reg_mid_phase_midnight-clocks"][0], e_reg_result_dict["reg_mid_phase_midnight-clocks"][0], f_reg_result_dict["reg_mid_phase_midnight-clocks"][0], g_reg_result_dict["reg_mid_phase_midnight-clocks"][0], h_reg_result_dict["reg_mid_phase_midnight-clocks"][0]]
# data_mid_clock = [a_reg_result_dict["reg_mid_phase_midnight-clocks"][1],b_reg_result_dict["reg_mid_phase_midnight-clocks"][1], c_reg_result_dict["reg_mid_phase_midnight-clocks"][1], d_reg_result_dict["reg_mid_phase_midnight-clocks"][1], e_reg_result_dict["reg_mid_phase_midnight-clocks"][1], f_reg_result_dict["reg_mid_phase_midnight-clocks"][1], g_reg_result_dict["reg_mid_phase_midnight-clocks"][1], h_reg_result_dict["reg_mid_phase_midnight-clocks"][1]]

# hist_data_mid = [data_mid_mid, data_mid_clock]
# fig_mid, ax_mid = plt.subplots()
# ax_mid.violinplot(hist_data_mid, showmedians = True)
# ax_mid.set_title('Betas only mid phase, z-scored data')
# ax_mid.set_xticks([1,2])
# ax_mid.set_xticklabels(["Midnight", "Clocks"])


# # late
# data_late_mid = [a_reg_result_dict["reg_late_phase_midnight-clocks"][0],b_reg_result_dict["reg_late_phase_midnight-clocks"][0], c_reg_result_dict["reg_late_phase_midnight-clocks"][0], d_reg_result_dict["reg_late_phase_midnight-clocks"][0], e_reg_result_dict["reg_late_phase_midnight-clocks"][0], f_reg_result_dict["reg_late_phase_midnight-clocks"][0], g_reg_result_dict["reg_late_phase_midnight-clocks"][0], h_reg_result_dict["reg_late_phase_midnight-clocks"][0]]
# data_late_clock = [a_reg_result_dict["reg_late_phase_midnight-clocks"][1],b_reg_result_dict["reg_late_phase_midnight-clocks"][1], c_reg_result_dict["reg_late_phase_midnight-clocks"][1], d_reg_result_dict["reg_late_phase_midnight-clocks"][1], e_reg_result_dict["reg_late_phase_midnight-clocks"][1], f_reg_result_dict["reg_late_phase_midnight-clocks"][1], g_reg_result_dict["reg_late_phase_midnight-clocks"][1], h_reg_result_dict["reg_late_phase_midnight-clocks"][1]]

# hist_data_late = [data_late_mid, data_late_clock]
# fig_late, ax_late = plt.subplots()
# ax_late.violinplot(hist_data_late, showmedians = True)
# ax_late.set_title('Betas only late phase, z-scored data')
# ax_late.set_xticks([1,2])
# ax_late.set_xticklabels(["Midnight", "Clocks"])


# # put back together
# data_reversedphase_mid = [a_reg_result_dict["reg_all_reversedphase_midnight-clocks"][0],b_reg_result_dict["reg_all_reversedphase_midnight-clocks"][0], c_reg_result_dict["reg_all_reversedphase_midnight-clocks"][0], d_reg_result_dict["reg_all_reversedphase_midnight-clocks"][0], e_reg_result_dict["reg_all_reversedphase_midnight-clocks"][0], f_reg_result_dict["reg_all_reversedphase_midnight-clocks"][0], g_reg_result_dict["reg_all_reversedphase_midnight-clocks"][0], h_reg_result_dict["reg_all_reversedphase_midnight-clocks"][0]]
# data_reversedphase_clock = [a_reg_result_dict["reg_all_reversedphase_midnight-clocks"][1],b_reg_result_dict["reg_all_reversedphase_midnight-clocks"][1], c_reg_result_dict["reg_all_reversedphase_midnight-clocks"][1], d_reg_result_dict["reg_all_reversedphase_midnight-clocks"][1], e_reg_result_dict["reg_all_reversedphase_midnight-clocks"][1], f_reg_result_dict["reg_all_reversedphase_midnight-clocks"][1], g_reg_result_dict["reg_all_reversedphase_midnight-clocks"][1], h_reg_result_dict["reg_all_reversedphase_midnight-clocks"][1]]

# hist_data_reversedphase = [data_reversedphase_mid, data_reversedphase_clock]
# fig_reversedphase, ax_reversedphase = plt.subplots()
# ax_reversedphase.violinplot(hist_data_reversedphase, showmedians = True)
# ax_reversedphase.set_title('Betas put back together in different order per phase, z-scored data')
# ax_reversedphase.set_xticks([1,2])
# ax_reversedphase.set_xticklabels(["Midnight", "Clocks"])


# # original 
# data_orig_mid = [a_reg_result_dict["reg_all_midnight-clocks-loc-phase"][0],b_reg_result_dict["reg_all_midnight-clocks-loc-phase"][0], c_reg_result_dict["reg_all_midnight-clocks-loc-phase"][0], d_reg_result_dict["reg_all_midnight-clocks-loc-phase"][0], e_reg_result_dict["reg_all_midnight-clocks-loc-phase"][0], f_reg_result_dict["reg_all_midnight-clocks-loc-phase"][0], g_reg_result_dict["reg_all_midnight-clocks-loc-phase"][0], h_reg_result_dict["reg_all_midnight-clocks-loc-phase"][0]]
# data_orig_clock = [a_reg_result_dict["reg_all_midnight-clocks-loc-phase"][1],b_reg_result_dict["reg_all_midnight-clocks-loc-phase"][1], c_reg_result_dict["reg_all_midnight-clocks-loc-phase"][1], d_reg_result_dict["reg_all_midnight-clocks-loc-phase"][1], e_reg_result_dict["reg_all_midnight-clocks-loc-phase"][1], f_reg_result_dict["reg_all_midnight-clocks-loc-phase"][1], g_reg_result_dict["reg_all_midnight-clocks-loc-phase"][1], h_reg_result_dict["reg_all_midnight-clocks-loc-phase"][1]]
# data_orig_loc = [a_reg_result_dict["reg_all_midnight-clocks-loc-phase"][2],b_reg_result_dict["reg_all_midnight-clocks-loc-phase"][2], c_reg_result_dict["reg_all_midnight-clocks-loc-phase"][2], d_reg_result_dict["reg_all_midnight-clocks-loc-phase"][2], e_reg_result_dict["reg_all_midnight-clocks-loc-phase"][2], f_reg_result_dict["reg_all_midnight-clocks-loc-phase"][2], g_reg_result_dict["reg_all_midnight-clocks-loc-phase"][2], h_reg_result_dict["reg_all_midnight-clocks-loc-phase"][2]]
# data_orig_phase = [a_reg_result_dict["reg_all_midnight-clocks-loc-phase"][3],b_reg_result_dict["reg_all_midnight-clocks-loc-phase"][3], c_reg_result_dict["reg_all_midnight-clocks-loc-phase"][3], d_reg_result_dict["reg_all_midnight-clocks-loc-phase"][3], e_reg_result_dict["reg_all_midnight-clocks-loc-phase"][3], f_reg_result_dict["reg_all_midnight-clocks-loc-phase"][3], g_reg_result_dict["reg_all_midnight-clocks-loc-phase"][3], h_reg_result_dict["reg_all_midnight-clocks-loc-phase"][3]]


# hist_data_orig = [data_orig_mid, data_orig_clock, data_orig_loc, data_orig_phase]
# fig_orig, ax_orig = plt.subplots()
# ax_orig.violinplot(hist_data_orig, showmedians = True)
# ax_orig.set_title('Betas original data, z-scored data')
# ax_orig.set_xticks([1,2,3,4])
# ax_orig.set_xticklabels(["Midnight", "Clocks", "Location", "Phase"])

# # build a mean between the early, middle, and late phase results
# mean_phase_mid = np.zeros(len(data_late_clock))
# mean_phase_clock = np.zeros(len(data_late_clock))
# for dataset in range(len(data_late_mid)):
#     mean_phase_mid[dataset] = np.mean((data_late_mid[dataset], data_early_mid[dataset], data_mid_mid[dataset]))
#     mean_phase_clock[dataset] = np.mean((data_late_clock[dataset], data_early_clock[dataset], data_mid_clock[dataset]))

# # hist_data_mean_phases = [mean_phase_mid, mean_phase_clock]
# # fig_mean_phases, ax_mean_phases = plt.subplots()
# # ax_mean_phases.violinplot(hist_data_mean_phases, showmedians = True, quantiles = [0.05, 0.95])
# # ax_mean_phases.set_title('Betas separated by phase and averaged, full phase on, neurons are z-scored')
# # ax_mean_phases.set_xticks([1,2])
# # ax_mean_phases.set_xticklabels(["Midnight", "Clocks"])


# hist_data_mean_phases = [mean_phase_mid, mean_phase_clock]
# fig_mean_phases, ax_mean_phases = plt.subplots()
# ax_mean_phases.boxplot(hist_data_mean_phases)

# ax_mean_phases.scatter(np.ones(len(hist_data_mean_phases[0])), hist_data_mean_phases[0])
# ax_mean_phases.scatter(np.ones(len(hist_data_mean_phases[1]))+1, hist_data_mean_phases[1])    

# ax_mean_phases.set_title('CONTINUOUS - Betas separated by phase and averaged, neurons are z-scored, no double tasks')
# ax_mean_phases.set_xticks([1,2])
# ax_mean_phases.set_xticklabels(["Midnight", "Clocks"])

# plt.axhline(0, color='grey', ls='dashed')

# # plot one violin plot per subject (i.e. 8 violins) where I visualize the variability across tasks per subject
# # but only if the distribution looks weird!!




# # # plot one violin plot per subject (i.e. 8 violins) where I visualize the variability across tasks per subject
# # # per contrast 

# # # mouse a, within task predictions for all recorded tasks.

# mean_contrasts_mouse_a[contr, task_no]
# set_of_columns = [mean_contrasts_mouse_a[0,:], mean_contrasts_mouse_a[4,:], mean_contrasts_mouse_a[5,:], mean_contrasts_mouse_a[6,:], mean_contrasts_mouse_a[3,:]]
# fig_mouse_a_contrasts, ax_mouse_a_contrasts = plt.subplots()
# ax_mouse_a_contrasts.boxplot(set_of_columns)
# ax_mouse_a_contrasts.scatter(np.ones(len(set_of_columns[0])), set_of_columns[0])
# ax_mouse_a_contrasts.scatter(np.ones(len(set_of_columns[1]))+1, set_of_columns[1])
# ax_mouse_a_contrasts.scatter(np.ones(len(set_of_columns[2]))+2, set_of_columns[2])
# ax_mouse_a_contrasts.scatter(np.ones(len(set_of_columns[3]))+3, set_of_columns[3])
# ax_mouse_a_contrasts.scatter(np.ones(len(set_of_columns[4]))+4, set_of_columns[4])
# ax_mouse_a_contrasts.set_xticklabels(["Only clocks", "Clocks-Midnight", "Clocks-loc", "Clocks-phase" , "only phase" ])
# ax_mouse_a_contrasts.set_xticks([1,2,3,4,5])
# ax_mouse_a_contrasts.set_title('mouse a, within task predictions for all recorded tasks')
# plt.axhline(0, color='grey', ls='dashed')


# mean_contrasts_mouse_b[contr, task_no]
# set_of_columns = [mean_contrasts_mouse_b[0,:], mean_contrasts_mouse_b[4,:], mean_contrasts_mouse_b[5,:], mean_contrasts_mouse_b[6,:], mean_contrasts_mouse_b[3,:]]
# fig_mouse_b_contrasts, ax_mouse_b_contrasts = plt.subplots()
# ax_mouse_b_contrasts.boxplot(set_of_columns)
# ax_mouse_b_contrasts.scatter(np.ones(len(set_of_columns[0])), set_of_columns[0])
# ax_mouse_b_contrasts.scatter(np.ones(len(set_of_columns[1]))+1, set_of_columns[1])
# ax_mouse_b_contrasts.scatter(np.ones(len(set_of_columns[2]))+2, set_of_columns[2])
# ax_mouse_b_contrasts.scatter(np.ones(len(set_of_columns[3]))+3, set_of_columns[3])
# ax_mouse_b_contrasts.scatter(np.ones(len(set_of_columns[4]))+4, set_of_columns[4])
# ax_mouse_b_contrasts.set_xticklabels(["Only clocks", "Clocks-Midnight", "Clocks-loc", "Clocks-phase", "only phase" ])
# ax_mouse_b_contrasts.set_xticks([1,2,3,4,5])
# ax_mouse_b_contrasts.set_title('mouse b, within task predictions for all recorded tasks')
# plt.axhline(0, color='grey', ls='dashed')



# # RUN THESE TWO!
# # compare what happens if I do more phase neurons
# mean_contrasts_mouse_a_11[contr, task_no]
# set_of_columns = [mean_contrasts_mouse_a_11[0,:], mean_contrasts_mouse_a_11[4,:], mean_contrasts_mouse_a_11[5,:], mean_contrasts_mouse_a_11[6,:], mean_contrasts_mouse_a_11[3,:]]
# fig_mouse_a_contrasts, ax_mouse_a_contrasts = plt.subplots()
# ax_mouse_a_contrasts.boxplot(set_of_columns)
# ax_mouse_a_contrasts.scatter(np.ones(len(set_of_columns[0])), set_of_columns[0])
# ax_mouse_a_contrasts.scatter(np.ones(len(set_of_columns[1]))+1, set_of_columns[1])
# ax_mouse_a_contrasts.scatter(np.ones(len(set_of_columns[2]))+2, set_of_columns[2])
# ax_mouse_a_contrasts.scatter(np.ones(len(set_of_columns[3]))+3, set_of_columns[3])
# ax_mouse_a_contrasts.scatter(np.ones(len(set_of_columns[4]))+4, set_of_columns[4])
# ax_mouse_a_contrasts.set_xticklabels(["Only clocks", "Clocks-Midnight", "Clocks-loc", "Clocks-phase" , "only phase" ])
# ax_mouse_a_contrasts.set_xticks([1,2,3,4,5])
# ax_mouse_a_contrasts.set_title('mouse a, within task predictions for all recorded tasks, 11 PHASE NEURONS!')
# plt.axhline(0, color='grey', ls='dashed')


# mean_contrasts_mouse_b_11[contr, task_no]
# set_of_columns = [mean_contrasts_mouse_b_11[0,:], mean_contrasts_mouse_b_11[4,:], mean_contrasts_mouse_b_11[5,:], mean_contrasts_mouse_b_11[6,:], mean_contrasts_mouse_b_11[3,:]]
# fig_mouse_b_contrasts, ax_mouse_b_contrasts = plt.subplots()
# ax_mouse_b_contrasts.boxplot(set_of_columns)
# ax_mouse_b_contrasts.scatter(np.ones(len(set_of_columns[0])), set_of_columns[0])
# ax_mouse_b_contrasts.scatter(np.ones(len(set_of_columns[1]))+1, set_of_columns[1])
# ax_mouse_b_contrasts.scatter(np.ones(len(set_of_columns[2]))+2, set_of_columns[2])
# ax_mouse_b_contrasts.scatter(np.ones(len(set_of_columns[3]))+3, set_of_columns[3])
# ax_mouse_b_contrasts.scatter(np.ones(len(set_of_columns[4]))+4, set_of_columns[4])
# ax_mouse_b_contrasts.set_xticklabels(["Only clocks", "Clocks-Midnight", "Clocks-loc", "Clocks-phase", "only phase" ])
# ax_mouse_b_contrasts.set_xticks([1,2,3,4,5])
# ax_mouse_b_contrasts.set_title('mouse b, within task predictions for all recorded tasks, 11 PHASE NEURONS!')
# plt.axhline(0, color='grey', ls='dashed')


# # RUN THESE TWO!
# # compare what happens if I do more phase neurons
# mean_contrasts_mouse_a[contr, task_no]
# set_of_columns = [mean_contrasts_mouse_a[0,:], mean_contrasts_mouse_a[1,:], mean_contrasts_mouse_a[2,:], mean_contrasts_mouse_a[3,:]]
# fig_mouse_a_contrasts, ax_mouse_a_contrasts = plt.subplots()
# ax_mouse_a_contrasts.boxplot(set_of_columns)
# ax_mouse_a_contrasts.scatter(np.ones(len(set_of_columns[0])), set_of_columns[0])
# ax_mouse_a_contrasts.scatter(np.ones(len(set_of_columns[1]))+1, set_of_columns[1])
# ax_mouse_a_contrasts.scatter(np.ones(len(set_of_columns[2]))+2, set_of_columns[2])
# ax_mouse_a_contrasts.scatter(np.ones(len(set_of_columns[3]))+3, set_of_columns[3])
# ax_mouse_a_contrasts.set_xticklabels(["Only clocks", "Only Midnight", "Only Loc", "Only phase"])
# ax_mouse_a_contrasts.set_xticks([1,2,3,4])
# ax_mouse_a_contrasts.set_title('mouse a, within task predictions for all recorded tasks, 3 PHASE NEURONS!')
# plt.axhline(0, color='grey', ls='dashed')


# mean_contrasts_mouse_b[contr, task_no]
# set_of_columns = [mean_contrasts_mouse_b[0,:], mean_contrasts_mouse_b[1,:], mean_contrasts_mouse_b[2,:], mean_contrasts_mouse_b[3,:]]
# fig_mouse_b_contrasts, ax_mouse_b_contrasts = plt.subplots()
# ax_mouse_b_contrasts.boxplot(set_of_columns)
# ax_mouse_b_contrasts.scatter(np.ones(len(set_of_columns[0])), set_of_columns[0])
# ax_mouse_b_contrasts.scatter(np.ones(len(set_of_columns[1]))+1, set_of_columns[1])
# ax_mouse_b_contrasts.scatter(np.ones(len(set_of_columns[2]))+2, set_of_columns[2])
# ax_mouse_b_contrasts.scatter(np.ones(len(set_of_columns[3]))+3, set_of_columns[3])
# ax_mouse_b_contrasts.set_xticklabels(["Only clocks", "Only Midnight", "Only Loc", "Only phase"])
# ax_mouse_b_contrasts.set_xticks([1,2,3,4])
# ax_mouse_b_contrasts.set_title('mouse b, within task predictions for all recorded tasks, 3 PHASE NEURONS!')
# plt.axhline(0, color='grey', ls='dashed')







# # Here I want to compare what the binning does to a within task regression.
# # to this means, plot the respective contrasts next to each other.

# contrast_matrix = ((1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1), (1,-1,0,0), (1, 0,-1,0), (1,0,0,-1), (0,1,-1,0), (0,1,0,-1), (0,0,1,-1))
# regression_mouse_a_3bins,  contrasts_mouse_a_3bins = mc.simulation.analyse_ephys.reg_per_task_config(a_rewards_configs, a_locations, a_neurons, a_timings, contrast_matrix, continuous = True, no_bins_per_state = 3, mouse_recday = 'me11_05122021_06122021')
# regression_mouse_a_10bins,  contrasts_mouse_a_10bins = mc.simulation.analyse_ephys.reg_per_task_config(a_rewards_configs, a_locations, a_neurons, a_timings, contrast_matrix, continuous = True, no_bins_per_state = 10, mouse_recday = 'me11_05122021_06122021')
# regression_mouse_a_50bins,  contrasts_mouse_a_50bins = mc.simulation.analyse_ephys.reg_per_task_config(a_rewards_configs, a_locations, a_neurons, a_timings, contrast_matrix, continuous = True, no_bins_per_state = 50, mouse_recday = 'me11_05122021_06122021')
# regression_mouse_a_no_bins,  contrasts_mouse_a_no_bins = mc.simulation.analyse_ephys.reg_per_task_config(a_rewards_configs, a_locations, a_neurons, a_timings, contrast_matrix, continuous = True, no_bins_per_state = 0, mouse_recday = 'me11_05122021_06122021')


# regression_mouse_b_3bins,  contrasts_mouse_b_3bins = mc.simulation.analyse_ephys.reg_per_task_config(b_rewards_configs, b_locations, b_neurons, b_timings, contrast_matrix, continuous = True, no_bins_per_state = 3, mouse_recday = 'me11_01122021_02122021')
# regression_mouse_b_10bins,  contrasts_mouse_b_10bins = mc.simulation.analyse_ephys.reg_per_task_config(b_rewards_configs, b_locations, b_neurons, b_timings, contrast_matrix, continuous = True, no_bins_per_state = 10, mouse_recday = 'me11_01122021_02122021')
# regression_mouse_b_50bins,  contrasts_mouse_b_50bins = mc.simulation.analyse_ephys.reg_per_task_config(b_rewards_configs, b_locations, b_neurons, b_timings, contrast_matrix, continuous = True, no_bins_per_state = 50, mouse_recday = 'me11_01122021_02122021')
# regression_mouse_b_no_bins,  contrasts_mouse_b_no_bins = mc.simulation.analyse_ephys.reg_per_task_config(b_rewards_configs, b_locations, b_neurons, b_timings, contrast_matrix, continuous = True, no_bins_per_state = 0, mouse_recday = 'me11_01122021_02122021')

# # for mouse a
# # for one task, I want to plot each of the 4 different bins for the 4 mean contrasts > 16 boxplots.
# for task_no in range(0, len(contrasts_mouse_a_3bins)):
#     data = []
#     for i in range(0,4):
#         data.append(contrasts_mouse_a_3bins[task_no][i])
#     for i in range(0,4):
#         data.append(contrasts_mouse_a_10bins[task_no][i])
#     for i in range(0,4):
#         data.append(contrasts_mouse_a_50bins[task_no][i])
#     for i in range(0,4):
#         data.append(contrasts_mouse_a_no_bins[task_no][i])
#     fig, ax = plt.subplots()
#     ax.boxplot(data)
#     for index, contrast in enumerate(data):
#         ax.scatter(np.ones(len(contrast))+index, contrast)
#     ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
#     plt.xticks(rotation = 45)
#     ax.set_xticklabels(["Clocks 3 bins","Midn 3 bins","Loc 3 bins","Phase 3 bins",  "Clocks 10 bins","Midn 10 bins", "Loc 10 bins","Phase 10 bins","Clocks 50 bins", "Midn 50 bins","Loc 50 bins","Phase 50 bins", "Clocks no bins",  "Midn no bins",  "Loc no bins",   "Phase no bins"])
#     plt.axhline(0, color='grey', ls='dashed')
#     plt.title(f"Comparing binning methods for MOUSE A and task no {task_no}")

    
# # for mouse b
# # for one task, I want to plot each of the 4 different bins for the 4 mean contrasts > 16 boxplots.
# for task_no in range(0, len(contrasts_mouse_b_3bins)):
#     data = []
#     for i in range(0,4):
#         data.append(contrasts_mouse_b_3bins[task_no][i])
#     for i in range(0,4):
#         data.append(contrasts_mouse_b_10bins[task_no][i])
#     for i in range(0,4):
#         data.append(contrasts_mouse_b_50bins[task_no][i])
#     for i in range(0,4):
#         data.append(contrasts_mouse_b_no_bins[task_no][i])
#     fig, ax = plt.subplots()
#     ax.boxplot(data)
#     for index, contrast in enumerate(data):
#         ax.scatter(np.ones(len(contrast))+index, contrast)
#     ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
#     plt.xticks(rotation = 45)
#     ax.set_xticklabels(["Clocks 3 bins","Midn 3 bins","Loc 3 bins","Phase 3 bins",  "Clocks 10 bins","Midn 10 bins", "Loc 10 bins","Phase 10 bins","Clocks 50 bins", "Midn 50 bins","Loc 50 bins","Phase 50 bins", "Clocks no bins",  "Midn no bins",  "Loc no bins",   "Phase no bins"])
#     plt.axhline(0, color='grey', ls='dashed')
#     plt.title(f"Comparing binning methods for MOUSE B and task no {task_no}")
  
    
# # look at the contrasts
# # for mouse b
# for task_no in range(0, len(contrasts_mouse_b_3bins)):
#     data = []
#     for i in range(0,3):
#         data.append(contrasts_mouse_b_3bins[task_no][i+4])
#     for i in range(0,3):
#         data.append(contrasts_mouse_b_10bins[task_no][i+4])
#     for i in range(0,3):
#         data.append(contrasts_mouse_b_50bins[task_no][i+4])
#     for i in range(0,3):
#         data.append(contrasts_mouse_b_no_bins[task_no][i+4])
#     fig, ax = plt.subplots()
#     ax.boxplot(data)
#     for index, contrast in enumerate(data):
#         ax.scatter(np.ones(len(contrast))+index, contrast)
#     ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11])
#     plt.xticks(rotation = 45)
#     ax.set_xticklabels(["Cl-Mid 3 bins","Cl-Loc 3 bins","Cl-Ph 3 bins",  "Cl-Mid 10 bins","Cl-Loc 10 bins", "Cl-Ph 10 bins",
#                         "Cl-Mid 50 bins", "Cl-Loc 50 bins","Cl-Ph 50 bins", "Cl-Mid no bins",  "Cl-Loc no bins",  "Cl-Ph no bins"])
#     plt.axhline(0, color='grey', ls='dashed')
#     plt.title(f"CONTRASTS - Comparing binning methods for MOUSE B and task no {task_no}")
  






# # COMPARE BINNING BETWEEN TASKS
# #  between-task regression based on single runs rather than averaging and then doing the regression.
# # possibility to try different binnings
# reg_between_mouse_a_3bin, contrast_between_mouse_a_3bin = mc.simulation.analyse_ephys.reg_between_tasks_singleruns(a_rewards_configs, a_locations, a_neurons, a_timings, contrast_matrix, mouse_recday = 'me11_05122021_06122021', continuous = True, no_bins_per_state = 3, ignore_double_tasks=1)
# reg_between_mouse_a_10bin, contrast_between_mouse_a_10bin = mc.simulation.analyse_ephys.reg_between_tasks_singleruns(a_rewards_configs, a_locations, a_neurons, a_timings, contrast_matrix, mouse_recday = 'me11_05122021_06122021', continuous = True, no_bins_per_state = 10, ignore_double_tasks=1)
# reg_between_mouse_a_30bin, contrast_between_mouse_a_30bin = mc.simulation.analyse_ephys.reg_between_tasks_singleruns(a_rewards_configs, a_locations, a_neurons, a_timings, contrast_matrix, mouse_recday = 'me11_05122021_06122021', continuous = True, no_bins_per_state = 30, ignore_double_tasks=1)

# reg_between_mouse_b_3bin, contrast_between_mouse_b_3bin = mc.simulation.analyse_ephys.reg_between_tasks_singleruns(b_rewards_configs, b_locations, b_neurons, b_timings, contrast_matrix, mouse_recday = 'me11_01122021_02122021', continuous = True, no_bins_per_state = 3, ignore_double_tasks=1)
# reg_between_mouse_b_10bin, contrast_between_mouse_b_10bin = mc.simulation.analyse_ephys.reg_between_tasks_singleruns(b_rewards_configs, b_locations, b_neurons, b_timings, contrast_matrix, mouse_recday = 'me11_01122021_02122021', continuous = True, no_bins_per_state = 10, ignore_double_tasks=1)
# reg_between_mouse_b_30bin, contrast_between_mouse_b_30bin = mc.simulation.analyse_ephys.reg_between_tasks_singleruns(b_rewards_configs, b_locations, b_neurons, b_timings, contrast_matrix, mouse_recday = 'me11_01122021_02122021', continuous = True, no_bins_per_state = 30, ignore_double_tasks=1)

# reg_between_mouse_c_3bin, contrast_between_mouse_c_3bin = mc.simulation.analyse_ephys.reg_between_tasks_singleruns(c_rewards_configs, c_locations, c_neurons, c_timings, contrast_matrix, mouse_recday = 'me10_09122021_10122021', continuous = True, no_bins_per_state = 3, ignore_double_tasks=1)
# reg_between_mouse_c_10bin, contrast_between_mouse_c_10bin = mc.simulation.analyse_ephys.reg_between_tasks_singleruns(c_rewards_configs, c_locations, c_neurons, c_timings, contrast_matrix, mouse_recday = 'me10_09122021_10122021', continuous = True, no_bins_per_state = 10, ignore_double_tasks=1)
# reg_between_mouse_c_30bin, contrast_between_mouse_c_30bin = mc.simulation.analyse_ephys.reg_between_tasks_singleruns(c_rewards_configs, c_locations, c_neurons, c_timings, contrast_matrix, mouse_recday = 'me10_09122021_10122021', continuous = True, no_bins_per_state = 30, ignore_double_tasks=1)

# reg_between_mouse_d_3bin, contrast_between_mouse_d_3bin = mc.simulation.analyse_ephys.reg_between_tasks_singleruns(d_rewards_configs, d_locations, d_neurons, d_timings, contrast_matrix, mouse_recday = 'me08_10092021_11092021', continuous = True, no_bins_per_state = 3, ignore_double_tasks=1)
# reg_between_mouse_d_10bin, contrast_between_mouse_d_10bin = mc.simulation.analyse_ephys.reg_between_tasks_singleruns(d_rewards_configs, d_locations, d_neurons, d_timings, contrast_matrix, mouse_recday = 'me08_10092021_11092021', continuous = True, no_bins_per_state = 10, ignore_double_tasks=1)
# reg_between_mouse_d_30bin, contrast_between_mouse_d_30bin = mc.simulation.analyse_ephys.reg_between_tasks_singleruns(d_rewards_configs, d_locations, d_neurons, d_timings, contrast_matrix, mouse_recday = 'me08_10092021_11092021', continuous = True, no_bins_per_state = 30, ignore_double_tasks=1)

# reg_between_mouse_e_3bin, contrast_between_mouse_e_3bin = mc.simulation.analyse_ephys.reg_between_tasks_singleruns(e_rewards_configs, e_locations, e_neurons, e_timings, contrast_matrix, mouse_recday = 'ah04_09122021_10122021', continuous = True, no_bins_per_state = 3, ignore_double_tasks=1)
# reg_between_mouse_e_10bin, contrast_between_mouse_e_10bin = mc.simulation.analyse_ephys.reg_between_tasks_singleruns(e_rewards_configs, e_locations, e_neurons, e_timings, contrast_matrix, mouse_recday = 'ah04_09122021_10122021', continuous = True, no_bins_per_state = 10, ignore_double_tasks=1)
# reg_between_mouse_e_30bin, contrast_between_mouse_e_30bin = mc.simulation.analyse_ephys.reg_between_tasks_singleruns(e_rewards_configs, e_locations, e_neurons, e_timings, contrast_matrix, mouse_recday = 'ah04_09122021_10122021', continuous = True, no_bins_per_state = 30, ignore_double_tasks=1)




# # look at what happens with more hase neurons: 11 PHASE NEURONS
# # COMPARE BETWEEN BINS IF I SPLIT BY PHASE

# # so for some reason the 3 binning doesn't work at all here. check this result!!
# # I just get crazy high values for phase.
# results_between_mouse_a_3bin_11 = mc.simulation.analyse_ephys.reg_between_tasks_singleruns(a_rewards_configs, a_locations, a_neurons, a_timings, contrast_matrix, mouse_recday = 'me11_05122021_06122021', continuous = True, no_bins_per_state = 3, ignore_double_tasks=1, split_by_phase= 1, number_phase_neurons= 11)



# results_between_mouse_a_10bin_11 = mc.simulation.analyse_ephys.reg_between_tasks_singleruns(a_rewards_configs, a_locations, a_neurons, a_timings, contrast_matrix, mouse_recday = 'me11_05122021_06122021', continuous = True, no_bins_per_state = 10, ignore_double_tasks=1, split_by_phase= 1, number_phase_neurons=11)


# results_between_mouse_a_30bin_11 = mc.simulation.analyse_ephys.reg_between_tasks_singleruns(a_rewards_configs, a_locations, a_neurons, a_timings, contrast_matrix, mouse_recday = 'me11_05122021_06122021', continuous = True, no_bins_per_state = 30, ignore_double_tasks=1, split_by_phase= 1, number_phase_neurons=11)


# # I want to know:
#     # 1. if there is a difference between the bins
#     # 2. how the betas change between early mid late
#     # 3. how it changes if I include phase or not
#     # 4. how it changes depending on the neuron-number in phase

# data = []
# #data.append(contrast_between_mouse_b_3bin[0,:],contrast_between_mouse_b_10bin[0,:],contrast_between_mouse_b_30bin[0,:])
# # contrast_between_mouse_a_3bin_11_early = results_between_mouse_a_3bin_11["contrast_early"]
# # contrast_between_mouse_a_3bin_11_mid = results_between_mouse_a_3bin_11["contrast_mid"]
# # contrast_between_mouse_a_3bin_11_late = results_between_mouse_a_3bin_11["contrast_late"]

# contrast_between_mouse_a_10bin_11_early = results_between_mouse_a_10bin_11["contrast_early"]
# contrast_between_mouse_a_10bin_11_mid = results_between_mouse_a_10bin_11["contrast_mid"]
# contrast_between_mouse_a_10bin_11_late = results_between_mouse_a_10bin_11["contrast_late"]

# # contrast_between_mouse_a_30bin_11_early = results_between_mouse_a_30bin_11["contrast_early"]
# # contrast_between_mouse_a_30bin_11_mid = results_between_mouse_a_30bin_11["contrast_mid"]
# # contrast_between_mouse_a_30bin_11_late = results_between_mouse_a_30bin_11["contrast_late"]



# for i in range(0,4):
#     #data.append(contrast_between_mouse_a_3bin_11_early[i,:])
#     data.append(contrast_between_mouse_a_10bin_11_early[i,:])
#     #data.append(contrast_between_mouse_a_30bin_11_early[i,:])
# for i in range(0,4):
#     #data.append(contrast_between_mouse_a_3bin_11_mid[i,:])
#     data.append(contrast_between_mouse_a_10bin_11_mid[i,:])
#     #data.append(contrast_between_mouse_a_30bin_11_mid[i,:])
# for i in range(0,4):
#     #data.append(contrast_between_mouse_a_3bin_11_late[i,:])
#     data.append(contrast_between_mouse_a_10bin_11_late[i,:])
#     #data.append(contrast_between_mouse_a_30bin_11_late[i,:])

# label_string_list = [ "clocks early", "midn early", "loc early", "phas early",
#                     "clocks mid", "midn mid","loc mid", "phas mid",
#                    "clocks late",  "midn late", "loc late",  "phas late"]
# label_tick_list = [0,1,2,3,4,5,6,7,8,9,10,11]
# title_string = "Split by phase, 10 bins, all model regs MOUSE A between tasks - 11 PHASE NEURONS"
# mc.simulation.analyse_ephys.plotting_hist_scat(data, label_string_list, label_tick_list, title_string)

# # fig, ax = plt.subplots()
# # ax.boxplot(data)
# # for index, contrast in enumerate(data):
# #     ax.scatter(np.ones(len(contrast))+index, contrast)
# # ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11])
# # plt.xticks(rotation = 45)
# # ax.set_xticklabels([ "clocks early", "midn early", "loc early", "phas early",
# #                     "clocks mid", "midn mid","loc mid", "phas mid",
# #                    "clocks late",  "midn late", "loc late",  "phas late"])

# # # ax.set_xticklabels([ "clocks 10 bins early", "clocks 30 bins early",  "midn 10 bins early", "midn 30 bins early", "loc 10 bins early", "loc 30 bins early", "phas 10 bins early", "phas 30 bins early",
# # #                     "clocks 10 bins mid", "clocks 30 bins mid","midn 10 bins mid", "midn 30 bins mid", "loc 10 bins mid", "loc 30 bins mid", "phas 10 bins mid", "phas 30 bins mid",
# # #                    "clocks 10 bins late", "clocks 30 bins late", "midn 10 bins late", "midn 30 bins late", "loc 10 bins late", "loc 30 bins late", "phas 10 bins late", "phas 30 bins late"])

# # # ax.set_xticklabels(["Cl-Mid 3 bins","Cl-Loc 3 bins","Cl-Ph 3 bins",  "Cl-Mid 10 bins","Cl-Loc 10 bins", "Cl-Ph 10 bins",
# # #                     "Cl-Mid 30 bins", "Cl-Loc 30 bins","Cl-Ph 30 bins"])
# # plt.axhline(0, color='grey', ls='dashed')
# # plt.title("Split by phase, 10 bins, all model regs MOUSE A between tasks - 11 PHASE NEURONS")
 



# # DO THE BETWEEN COMPARISON FOR 11 NEURONS and DIFFERENT BINS IF I DO  NOT SPLIT PHASE.
# results_between_mouse_a_3bin_11_one = mc.simulation.analyse_ephys.reg_between_tasks_singleruns(a_rewards_configs, a_locations, a_neurons, a_timings, contrast_matrix, mouse_recday = 'me11_05122021_06122021', continuous = True, no_bins_per_state = 3, ignore_double_tasks=1, split_by_phase= 0, number_phase_neurons= 11)
# results_between_mouse_a_10bin_11_one = mc.simulation.analyse_ephys.reg_between_tasks_singleruns(a_rewards_configs, a_locations, a_neurons, a_timings, contrast_matrix, mouse_recday = 'me11_05122021_06122021', continuous = True, no_bins_per_state = 10, ignore_double_tasks=1, split_by_phase= 0, number_phase_neurons=11)
# results_between_mouse_a_30bin_11_one = mc.simulation.analyse_ephys.reg_between_tasks_singleruns(a_rewards_configs, a_locations, a_neurons, a_timings, contrast_matrix, mouse_recday = 'me11_05122021_06122021', continuous = True, no_bins_per_state = 30, ignore_double_tasks=1, split_by_phase= 0, number_phase_neurons=11)


# # I want to know:
#     # 1. if there is a difference between the bins
#     # 2. how the betas change between early mid late
#     # 3. how it changes if I include phase or not
#     # 4. how it changes depending on the neuron-number in phase

# data = []
# #data.append(contrast_between_mouse_b_3bin[0,:],contrast_between_mouse_b_10bin[0,:],contrast_between_mouse_b_30bin[0,:])
# contrast_between_mouse_a_3bin_11_one = results_between_mouse_a_3bin_11_one["contrast_results"]
# contrast_between_mouse_a_10bin_11_one = results_between_mouse_a_10bin_11_one["contrast_results"]
# contrast_between_mouse_a_30bin_11_one = results_between_mouse_a_30bin_11_one["contrast_results"]


# for i in range(0,4):
#     data.append(contrast_between_mouse_a_3bin_11_one[i,:])
#     data.append(contrast_between_mouse_a_10bin_11_one[i,:])
#     data.append(contrast_between_mouse_a_30bin_11_one[i,:])

    
# fig, ax = plt.subplots()
# ax.boxplot(data)
# for index, contrast in enumerate(data):
#     ax.scatter(np.ones(len(contrast))+index, contrast)
# ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11])
# plt.xticks(rotation = 45)
# ax.set_xticklabels(["clocks 3 bins", "clocks 10 bins", "clocks 30 bins", "midn 3 bins", "midn 10 bins", "midn 30 bin", "loc 3 bin", "loc 10 bin", "loc 30 bin", "phas 3 bin", "phas 10 bin", "phas 30 bins"])
# # ax.set_xticklabels(["Cl-Mid 3 bins","Cl-Loc 3 bins","Cl-Ph 3 bins",  "Cl-Mid 10 bins","Cl-Loc 10 bins", "Cl-Ph 10 bins",
# #                     "Cl-Mid 30 bins", "Cl-Loc 30 bins","Cl-Ph 30 bins"])
# plt.axhline(0, color='grey', ls='dashed')
# plt.title("Binning comparison plus split by phase with all model regs MOUSE A between tasks - 11 PHASE NEURONS")
 


# # DO THE BETWEEN COMPARISON FOR 3 NEURONS and DIFFERENT BINS IF I DO  NOT SPLIT PHASE.
# results_between_mouse_a_3bin_3_one = mc.simulation.analyse_ephys.reg_between_tasks_singleruns(a_rewards_configs, a_locations, a_neurons, a_timings, contrast_matrix, mouse_recday = 'me11_05122021_06122021', continuous = True, no_bins_per_state = 3, ignore_double_tasks=1, split_by_phase= 0, number_phase_neurons= 3)
# results_between_mouse_a_10bin_3_one = mc.simulation.analyse_ephys.reg_between_tasks_singleruns(a_rewards_configs, a_locations, a_neurons, a_timings, contrast_matrix, mouse_recday = 'me11_05122021_06122021', continuous = True, no_bins_per_state = 10, ignore_double_tasks=1, split_by_phase= 0, number_phase_neurons=3)
# results_between_mouse_a_30bin_3_one = mc.simulation.analyse_ephys.reg_between_tasks_singleruns(a_rewards_configs, a_locations, a_neurons, a_timings, contrast_matrix, mouse_recday = 'me11_05122021_06122021', continuous = True, no_bins_per_state = 30, ignore_double_tasks=1, split_by_phase= 0, number_phase_neurons=3)


# # I want to know:
#     # 1. if there is a difference between the bins
#     # 2. how the betas change between early mid late
#     # 3. how it changes if I include phase or not
#     # 4. how it changes depending on the neuron-number in phase

# data = []
# #data.append(contrast_between_mouse_b_3bin[0,:],contrast_between_mouse_b_10bin[0,:],contrast_between_mouse_b_30bin[0,:])
# contrast_between_mouse_a_3bin_3_one = results_between_mouse_a_3bin_3_one["contrast_results"]
# contrast_between_mouse_a_10bin_3_one = results_between_mouse_a_10bin_3_one["contrast_results"]
# contrast_between_mouse_a_30bin_3_one = results_between_mouse_a_30bin_3_one["contrast_results"]


# for i in range(0,4):
#     data.append(contrast_between_mouse_a_3bin_3_one[i,:])
#     data.append(contrast_between_mouse_a_10bin_3_one[i,:])
#     data.append(contrast_between_mouse_a_30bin_3_one[i,:])

    
# fig, ax = plt.subplots()
# ax.boxplot(data)
# for index, contrast in enumerate(data):
#     ax.scatter(np.ones(len(contrast))+index, contrast)
# ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11])
# plt.xticks(rotation = 45)
# ax.set_xticklabels(["clocks 3 bins", "clocks 10 bins", "clocks 30 bins", "midn 3 bins", "midn 10 bins", "midn 30 bin", "loc 3 bin", "loc 10 bin", "loc 30 bin", "phas 3 bin", "phas 10 bin", "phas 30 bins"])
# # ax.set_xticklabels(["Cl-Mid 3 bins","Cl-Loc 3 bins","Cl-Ph 3 bins",  "Cl-Mid 10 bins","Cl-Loc 10 bins", "Cl-Ph 10 bins",
# #                     "Cl-Mid 30 bins", "Cl-Loc 30 bins","Cl-Ph 30 bins"])
# plt.axhline(0, color='grey', ls='dashed')
# plt.title("Binning comparison plus split by phase with all model regs MOUSE A between tasks - 3 PHASE NEURONS")
 





# # CHECK WHAT CHANGES IF I ONLY TAKE CERTAIN NEURONS.
# # here I onlye select the anchored neurons. the midnight and clocks model should get a lot better
# # in predicting the data.

# # midnight = Location + Anchor are true
# # clocks = Anchors is true

# a_anchor_neurons = []
# a_anchor_mask = a_cells[:,-1]

# for task, neurons in enumerate(a_neurons):
#     a_anchor_neurons.append(neurons[a_anchor_mask, :])
    
# results_between_mouse_a_10bin_11_anchor = mc.simulation.analyse_ephys.reg_between_tasks_singleruns(a_rewards_configs, a_locations, a_anchor_neurons, a_timings, contrast_matrix, mouse_recday = 'me11_05122021_06122021', continuous = True, no_bins_per_state = 10, ignore_double_tasks=1, split_by_phase= 1, number_phase_neurons=11)

# data = []

# contrast_between_mouse_a_10bin_11_early_anchor = results_between_mouse_a_10bin_11_anchor["contrast_early"]
# contrast_between_mouse_a_10bin_11_mid_anchor = results_between_mouse_a_10bin_11_anchor["contrast_mid"]
# contrast_between_mouse_a_10bin_11_late_anchor = results_between_mouse_a_10bin_11_anchor["contrast_late"]


# for i in range(0,4):
#     #data.append(contrast_between_mouse_a_3bin_11_early[i,:])
#     data.append(contrast_between_mouse_a_10bin_11_early_anchor[i,:])
#     #data.append(contrast_between_mouse_a_30bin_11_early[i,:])
# for i in range(0,4):
#     #data.append(contrast_between_mouse_a_3bin_11_mid[i,:])
#     data.append(contrast_between_mouse_a_10bin_11_mid_anchor[i,:])
#     #data.append(contrast_between_mouse_a_30bin_11_mid[i,:])
# for i in range(0,4):
#     #data.append(contrast_between_mouse_a_3bin_11_late[i,:])
#     data.append(contrast_between_mouse_a_10bin_11_late_anchor[i,:])
#     #data.append(contrast_between_mouse_a_30bin_11_late[i,:])
    
# fig, ax = plt.subplots()
# ax.boxplot(data)
# for index, contrast in enumerate(data):
#     ax.scatter(np.ones(len(contrast))+index, contrast)
# ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11])
# plt.xticks(rotation = 45)
# ax.set_xticklabels([ "clocks early", "midn early", "loc early", "phas early",
#                     "clocks mid", "midn mid","loc mid", "phas mid",
#                    "clocks late",  "midn late", "loc late",  "phas late"])

# # ax.set_xticklabels([ "clocks 10 bins early", "clocks 30 bins early",  "midn 10 bins early", "midn 30 bins early", "loc 10 bins early", "loc 30 bins early", "phas 10 bins early", "phas 30 bins early",
# #                     "clocks 10 bins mid", "clocks 30 bins mid","midn 10 bins mid", "midn 30 bins mid", "loc 10 bins mid", "loc 30 bins mid", "phas 10 bins mid", "phas 30 bins mid",
# #                    "clocks 10 bins late", "clocks 30 bins late", "midn 10 bins late", "midn 30 bins late", "loc 10 bins late", "loc 30 bins late", "phas 10 bins late", "phas 30 bins late"])

# # ax.set_xticklabels(["Cl-Mid 3 bins","Cl-Loc 3 bins","Cl-Ph 3 bins",  "Cl-Mid 10 bins","Cl-Loc 10 bins", "Cl-Ph 10 bins",
# #                     "Cl-Mid 30 bins", "Cl-Loc 30 bins","Cl-Ph 30 bins"])
# plt.axhline(0, color='grey', ls='dashed')
# plt.title("Split by phase, 10 bins, ONLY ANCHORED NEURONS! all model regs MOUSE A between tasks - 11 PHASE NEURONS")
 










    
# # look at the contrasts
# # for mouse a
# data = []
# #data.append(contrast_between_mouse_b_3bin[0,:],contrast_between_mouse_b_10bin[0,:],contrast_between_mouse_b_30bin[0,:])
# for i in range(0,3):
#     data.append(contrast_between_mouse_a_3bin[i+4,:])
# for i in range(0,3):
#     data.append(contrast_between_mouse_a_10bin[i+4,:])
# for i in range(0,3):
#     data.append(contrast_between_mouse_a_30bin[i+4,:])
# fig, ax = plt.subplots()
# ax.boxplot(data)
# for index, contrast in enumerate(data):
#     ax.scatter(np.ones(len(contrast))+index, contrast)
# ax.set_xticks([0,1,2,3,4,5,6,7,8])
# plt.xticks(rotation = 45)
# ax.set_xticklabels(["Cl-Mid 3 bins","Cl-Loc 3 bins","Cl-Ph 3 bins",  "Cl-Mid 10 bins","Cl-Loc 10 bins", "Cl-Ph 10 bins",
#                     "Cl-Mid 30 bins", "Cl-Loc 30 bins","Cl-Ph 30 bins"])
# plt.axhline(0, color='grey', ls='dashed')
# plt.title(f"CONTRASTS - Comparing binning methods for MOUSE A between tasks per run")
 

# # for mouse b
# data = []
# #data.append(contrast_between_mouse_b_3bin[0,:],contrast_between_mouse_b_10bin[0,:],contrast_between_mouse_b_30bin[0,:])
# for i in range(0,3):
#     data.append(contrast_between_mouse_b_3bin[i+4,:])
# for i in range(0,3):
#     data.append(contrast_between_mouse_b_10bin[i+4,:])
# for i in range(0,3):
#     data.append(contrast_between_mouse_b_30bin[i+4,:])
# fig, ax = plt.subplots()
# ax.boxplot(data)
# for index, contrast in enumerate(data):
#     ax.scatter(np.ones(len(contrast))+index, contrast)
# ax.set_xticks([0,1,2,3,4,5,6,7,8])
# plt.xticks(rotation = 45)
# ax.set_xticklabels(["Cl-Mid 3 bins","Cl-Loc 3 bins","Cl-Ph 3 bins",  "Cl-Mid 10 bins","Cl-Loc 10 bins", "Cl-Ph 10 bins",
#                     "Cl-Mid 30 bins", "Cl-Loc 30 bins","Cl-Ph 30 bins"])
# plt.axhline(0, color='grey', ls='dashed')
# plt.title(f"CONTRASTS - Comparing binning methods for MOUSE B between tasks per run")
  

# # for mouse c
# data = []
# #data.append(contrast_between_mouse_b_3bin[0,:],contrast_between_mouse_b_10bin[0,:],contrast_between_mouse_b_30bin[0,:])
# for i in range(0,3):
#     data.append(contrast_between_mouse_c_3bin[i+4,:])
# for i in range(0,3):
#     data.append(contrast_between_mouse_c_10bin[i+4,:])
# for i in range(0,3):
#     data.append(contrast_between_mouse_c_30bin[i+4,:])
# fig, ax = plt.subplots()
# ax.boxplot(data)
# for index, contrast in enumerate(data):
#     ax.scatter(np.ones(len(contrast))+index, contrast)
# ax.set_xticks([1,2,3,4,5,6,7,8,9])
# plt.xticks(rotation = 45)
# ax.set_xticklabels(["Cl-Mid 3 bins","Cl-Loc 3 bins","Cl-Ph 3 bins",  "Cl-Mid 10 bins","Cl-Loc 10 bins", "Cl-Ph 10 bins",
#                     "Cl-Mid 30 bins", "Cl-Loc 30 bins","Cl-Ph 30 bins"])
# plt.axhline(0, color='grey', ls='dashed')
# plt.title(f"CONTRASTS - Comparing binning methods for MOUSE C between tasks per run")


# # for mouse d
# data = []
# #data.append(contrast_between_mouse_b_3bin[0,:],contrast_between_mouse_b_10bin[0,:],contrast_between_mouse_b_30bin[0,:])
# for i in range(0,3):
#     data.append(contrast_between_mouse_d_3bin[i+4,:])
# for i in range(0,3):
#     data.append(contrast_between_mouse_d_10bin[i+4,:])
# for i in range(0,3):
#     data.append(contrast_between_mouse_d_30bin[i+4,:])
# fig, ax = plt.subplots()
# ax.boxplot(data)
# for index, contrast in enumerate(data):
#     ax.scatter(np.ones(len(contrast))+index, contrast)
# ax.set_xticks([0,1,2,3,4,5,6,7,8])
# plt.xticks(rotation = 45)
# ax.set_xticklabels(["Cl-Mid 3 bins","Cl-Loc 3 bins","Cl-Ph 3 bins",  "Cl-Mid 10 bins","Cl-Loc 10 bins", "Cl-Ph 10 bins",
#                     "Cl-Mid 30 bins", "Cl-Loc 30 bins","Cl-Ph 30 bins"])
# plt.axhline(0, color='grey', ls='dashed')
# plt.title(f"CONTRASTS - Comparing binning methods for MOUSE D between tasks per run")

# # for mouse e
# data = []
# #data.append(contrast_between_mouse_b_3bin[0,:],contrast_between_mouse_b_10bin[0,:],contrast_between_mouse_b_30bin[0,:])
# for i in range(0,3):
#     data.append(contrast_between_mouse_e_3bin[i+4,:])
# for i in range(0,3):
#     data.append(contrast_between_mouse_e_10bin[i+4,:])
# for i in range(0,3):
#     data.append(contrast_between_mouse_e_30bin[i+4,:])
# fig, ax = plt.subplots()
# ax.boxplot(data)
# for index, contrast in enumerate(data):
#     ax.scatter(np.ones(len(contrast))+index, contrast)
# ax.set_xticks([0,1,2,3,4,5,6,7,8])
# plt.xticks(rotation = 45)
# ax.set_xticklabels(["Cl-Mid 3 bins","Cl-Loc 3 bins","Cl-Ph 3 bins",  "Cl-Mid 10 bins","Cl-Loc 10 bins", "Cl-Ph 10 bins",
#                     "Cl-Mid 30 bins", "Cl-Loc 30 bins","Cl-Ph 30 bins"])
# plt.axhline(0, color='grey', ls='dashed')
# plt.title(f"CONTRASTS - Comparing binning methods for MOUSE E between tasks per run")



# # Since the phase-regressor is always way stronger for the between-data analysis, 
# # next step is to make phase irrelavant.
# # it seemed as if the 30-binning was quite a good choice. This will also allow me 
# # to more or less get rid of phase.
# results_mouse_a_6bins = mc.simulation.analyse_ephys.reg_between_tasks_singleruns(a_rewards_configs, a_locations, a_neurons, a_timings, contrast_matrix, continuous = True, no_bins_per_state = 6, mouse_recday = 'me11_05122021_06122021', split_by_phase = 1)






# # # data_contr_1 = [mean_contrasts_mouse_a[0,:], mean_contrasts_mouse_b[0,:], mean_contrasts_mouse_c[0,:], mean_contrasts_mouse_d[0,:], mean_contrasts_mouse_e[0,:], mean_contrasts_mouse_f[0,:], mean_contrasts_mouse_g[0,:], mean_contrasts_mouse_h[0,:]]
# # # data_contr_2 = [mean_contrasts_mouse_a[1,:], mean_contrasts_mouse_b[1,:], mean_contrasts_mouse_c[1,:], mean_contrasts_mouse_d[1,:], mean_contrasts_mouse_e[1,:], mean_contrasts_mouse_f[1,:], mean_contrasts_mouse_g[1,:], mean_contrasts_mouse_h[1,:]]
# # # data_contr_3 = [mean_contrasts_mouse_a[2,:], mean_contrasts_mouse_b[2,:], mean_contrasts_mouse_c[2,:], mean_contrasts_mouse_d[2,:], mean_contrasts_mouse_e[2,:], mean_contrasts_mouse_f[2,:], mean_contrasts_mouse_g[2,:], mean_contrasts_mouse_h[2,:]]



# # contr_1_mean_clock = [mean_contrasts_mouse_a[0,:], mean_contrasts_mouse_b[0,:],  mean_contrasts_mouse_g[0,:]]
# # contr_2_mean_midnight = [mean_contrasts_mouse_a[1,:], mean_contrasts_mouse_b[1,:], mean_contrasts_mouse_g[1,:]]
# # contr_3_mean_location = [mean_contrasts_mouse_a[2,:], mean_contrasts_mouse_b[2,:], mean_contrasts_mouse_g[2,:]]
# # contr_4_mean_phase = [mean_contrasts_mouse_a[3,:], mean_contrasts_mouse_b[3,:], mean_contrasts_mouse_g[3,:]]

# # contr_5_clocks_midnight = [mean_contrasts_mouse_a[4,:], mean_contrasts_mouse_b[4,:], mean_contrasts_mouse_g[4,:]]
# # contr_6_clocks_location = [mean_contrasts_mouse_a[5,:], mean_contrasts_mouse_b[5,:], mean_contrasts_mouse_g[5,:]]
# # contr_7_clocks_phase = [mean_contrasts_mouse_a[6,:], mean_contrasts_mouse_b[6,:], mean_contrasts_mouse_g[6,:]]


                                         
# # fig_one, ax_one = joypy.joyplot(contr_1_mean_clock, linewidth = 0.05, overlap = 2, colormap = cm.summer_r, x_range = [-0.13,0.13], fade = True)
# # plt.title('Mean clocks, distr. across tasks per mouse [1 0 0 0]')
# # plt.show()

# # fig_two, ax_two = joypy.joyplot(contr_2_mean_midnight, linewidth = 0.05, overlap = 2, colormap = cm.summer_r, x_range = [-0.13,0.13], fade = True)
# # plt.title('Mean midnight, distr. across tasks per mouse [0 1 0 0]')
# # plt.show()

# # fig_three, ax_three = joypy.joyplot(contr_3_mean_location, linewidth = 0.05, overlap = 2, colormap = cm.summer_r, x_range = [-0.13,0.13], fade = True)
# # plt.title('Mean location, distr. across tasks per mouse [0 0 1 0]')
# # plt.show()

# # fig_four, ax_four = joypy.joyplot(contr_4_mean_phase, linewidth = 0.05, overlap = 2, colormap = cm.summer_r, x_range = [-0.13,0.13], fade = True)
# # plt.title('Mean phase, distr. across tasks per mouse [0 0 0 1]')
# # plt.show()


# # #########

# # fig_contr_one, ax_contr_one = joypy.joyplot(contr_7_clocks_phase, linewidth = 0.05, overlap = 2, colormap = cm.summer_r, x_range = [-0.3,0.3], fade = True)
# # plt.title('Clocks-phase, distr. across tasks per mouse [1 0 0 -1]')
# # plt.show()

# # fig_contr_two, axcontr__two = joypy.joyplot(contr_5_clocks_midnight, linewidth = 0.05, overlap = 2, colormap = cm.summer_r, x_range = [-0.3,0.3], fade = True)
# # plt.title('Clocks-midnight contrast, distr. across tasks per mouse [1 -1 0 0]')
# # plt.show()

# # fig_contr_three, ax_contr_three = joypy.joyplot(contr_6_clocks_location, linewidth = 0.05, overlap = 2, colormap = cm.summer_r, x_range = [-0.3,0.3], fade = True)
# # plt.title('Clocks-location contrast, distr. across tasks per mouse [1 0 -1 0]')
# # plt.show()





# # # plotting the mean across runs for each task for each beta separateyl (1 0 0 0,...)
# # x1 = np.array(mean_beta_clocks_a)
# # x2 = np.array(mean_beta_midnight_a)
# # x3 = np.array(mean_beta_locations_a)
# # x4 = np.array(mean_beta_phase_a)
# # hist_data = [x1, x2, x3, x4]

# # # group_labels = ['Clocks', 'Midnight', 'Location']
# # # colors = ['#A56CC1', '#A6ACEC', '#63F5EF']

# # # # Create distplot with curve_type set to 'normal'
# # # fig = ff.create_distplot(hist_data, group_labels, colors=colors,
# # #                          bin_size=.2, show_rug=False)

# # # # Add title
# # # fig.update_layout(title_text='Mouse 1, mean beta per task configuration')
# # # fig.show()

# # fig, ax = plt.subplots()
# # ax.violinplot(hist_data, showmedians = True)
# # ax.set_title('Beta weights for a single mouse across task configs')
# # ax.set_xticks([1,2,3,4])
# # ax.set_xticklabels(["Clocks", "Midnight", "Location", "Phase"])


# # contr_5_clocks_midnight = np.concatenate((mean_contrasts_mouse_a[4,:], mean_contrasts_mouse_b[4,:], mean_contrasts_mouse_g[4,:]), axis = None)
# # contr_6_clocks_location = np.concatenate((mean_contrasts_mouse_a[5,:], mean_contrasts_mouse_b[5,:], mean_contrasts_mouse_g[5,:]), axis = None)
# # contr_7_clocks_phase = np.concatenate((mean_contrasts_mouse_a[6,:], mean_contrasts_mouse_b[6,:], mean_contrasts_mouse_g[6,:]), axis = None)

# # # tstat of these values
# # t_stat_contr_5 = scipy.stats.ttest_1samp(contr_5_clocks_midnight, 0)


# # hist_data_contrasts = [contr_5_clocks_midnight, contr_6_clocks_location, contr_7_clocks_phase]
# # fig_con, ax_con = plt.subplots()
# # ax_con.violinplot(hist_data_contrasts, showmedians = True)
# # ax_con.set_title('Contrasts for 3 mice across tasks and runs')
# # ax_con.set_xticks([1,2,3])
# # ax_con.set_xticklabels(["Clocks - Midnight", "Clocks-Location", "Clocks-Phase"])



# #
# # Part 4: Group level analysis
# #



    
# # GROUP LEVEL ANALYSIS next.
# # problem: there might be outliers!!
# # coorperate a breakpoint if beta > 100 or so to check what is going wrong with the GLM in this case



    
#     # 1. subject level.
#     # for every mouse and every run, compute a GLM with my 3 regressors.
#     # 2. compute contrasts:
#     # I want to know: [0 0 1], [0 1 0], [1 0 0] and [-1 1 0], [0 -1 1], ....
#     # (every regressor at its own, and the cotnrast between 2 betas (MRI: PEs) which is 
#     # 1 minus the other)
#     # take all of these values and average 
#     #        1. across runs within one task config
#     #       2. across task configs
#     # Finally, you end up with 9 betas (MRI: COPEs) for every mouse (contrasts)
#     # 3. Group level:
#         # compute a random effects GLM for every of the contrasts, using
#         # each mouse-beta as an input
    
    
    
#     # next step: Stats! > group statistics?
#     # > multiple runs? use the regressor model since there are different fields the mouse runs on? 
#     # across tasks?? 
#     # also check the correlation values, independent from the other regressors
    
#     # before: within a mouse > fixed effects (+averaging betas) to compare runs 
    
    
#     # last step: check ou FSL FEAT -> compare betas across mice with random effects if 
#     # thats possible with only 8 mice, otherwise fixed effects 
    
    
    
#     # potentially, also have a look at a second thing:
#         # concatenate all trials across task cofngis
#         # then reduce the size of the data to steps instead of ms
#         # by using the step-regressors from the fMRI model (or the other way around)
#     # afterwards, follow the same group-stats and contrasts
#     # these contrasts are probably even more significant and will be more like my fmRI analyssi
    




