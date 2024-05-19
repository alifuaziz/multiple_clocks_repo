# # Python libraries
# from tqdm import tqdm
# import numpy as np
# import matplotlib.pyplot as plt
# import pickle
# import sys
# import random
# import os
# from pathlib import Path
# from joblib import Parallel, delayed

# # RSA specific libraries
# import nibabel as nib
# from nilearn.image import load_img
# import rsatoolbox.rdm as rsr
# import rsatoolbox.vis as vis
# import rsatoolbox
# from rsatoolbox.rdm import RDMs
# from rsatoolbox.util.searchlight import get_volume_searchlight, get_searchlight_RDMs

# # mc imports
# import mc.analyse.analyse_MRI_behav     as analyse_MRI_behav
# import mc.analyse.calc                  as calc
# import mc.replay_analysis.functions.model_rdms as model_rdm_functions


# # Constants
# REGRESSION_VERSION = '01' 
# RDM_VERSION        = '01-2' 
# SUB_NO = '01'
# USE_PREVIOUS_SEARCHLIGHTS = True  # Searchlights are loaded from file
# USE_PREVIOUS_DATA_RDM     = False # Data RDMs are loaded from file
# VISUALISE_RDMS            = False # Visualise the data RDMs
# REMOVE_AUTOCORR           = True  # Remove autocorrelations from the data RDMs, else cross-correlate
# EVS_TYPE                  = "instruction_period"     # The parts of the BOLD signal that are used for the RSA

# # Paths
# data_folder = Path('/Users/student/PycharmProjects/data')

# # Subjects to be analysed
# sub: str = f"sub-{SUB_NO}"
# #subjects = ['sub-01']
# task_halves: list = ['1', '2']



# print(f"Now running RSA for RDM version {RDM_VERSION} based on subj GLM {REGRESSION_VERSION} for subj {SUB_NO}")

# # get the list of the models to be analysed
# models_I_want: list = analyse_MRI_behav.models_I_want(RDM_VERSION)

# # based on GLM, get the number of conditions in the RDM
# no_RDM_conditions: int = analyse_MRI_behav.get_no_RDM_conditions(RDM_VERSION)



# subject_directory = data_folder / 'derivatives' / sub
# print(subject_directory)
# if os.path.isdir(subject_directory):
#     print("Running on laptop.")


# RDM_dir = subject_directory / 'beh' / f"RDMs_{RDM_VERSION}_glmbase_{REGRESSION_VERSION}"


# # Make the RDM_dir if it doesn't exist
# if not RDM_dir.exists():
#     RDM_dir.mkdir(parents=True)
# results_dir = subject_directory / 'func' / f"RSA_{RDM_VERSION}_glmbase_{REGRESSION_VERSION}"

# # Make the results_dir if it doesn't exist
# # if not results_dir.exists():
# #     results_dir.mkdir(parents=True)
# #     os.makedirs(f"{results_dir}/results")
# # results_dir = f"{subject_directory}/func/RSA_{RDM_VERSION}_glmbase_{REGRESSION_VERSION}/results" 
# # if os.path.exists(results_dir):
# #     # move pre-existing files into a different folder.
# #     analyse_MRI_behav.move_files_to_subfolder(results_dir)
# # get a reference image to later project the results onto. This is usually
# # example_func from half 1, as this is where the data is corrected to.

# # results_dir = results_dir / 'results' 
# ref_img = load_img( subject_directory / 'func' / 'preproc_clean_01.feat'/ 'example_func.nii.gz' )

# # load the file which defines the order of the model RDMs, and hence the data RDMs
# with open(f"{RDM_dir}/sorted_keys-model_RDMs.pkl", 'rb') as file:
#     sorted_keys = pickle.load(file)
# with open(f"{RDM_dir}/sorted_regs.pkl", 'rb') as file:
#     reg_keys = pickle.load(file)
# # also store 2 dictionaries of the EVs




# REGRESSION_VERSION = analyse_MRI_behav.preprocess_regression_version(REGRESSION_VERSION)

# # create dictionary of paths to the EVs for each half (split) of the task
# # first half
# EV_path_dict_01 = analyse_MRI_behav.get_EV_dict(
#     subject_directory, REGRESSION_VERSION
#     )
# # second half
# EV_path_dict_02 = analyse_MRI_behav.get_EV_dict(
#     subject_directory, REGRESSION_VERSION
#     )


# # Step 1: get the searchlights

# # Load the functional mask of the subject (i.e. the brain mask of only the computation parts of the brain)
# mask = load_img(f"{subject_directory}/anat/{sub}_T1w_noCSF_brain_mask_bin_func_01.nii.gz")

# # Get the searchlights
# centers, neighbors = model_rdm_functions.get_searchlights(
#     mask = mask,
#     radius = 3, 
#     threshold = 0.5,
#     USE_PREVIOUS_SEARCHLIGHTS = USE_PREVIOUS_SEARCHLIGHTS,
#     NEIGHBORS_PATH=f"{RDM_dir}/searchlight_neighbors.pkl",
#     CENTERS_PATH=f"{RDM_dir}/searchlight_centers.pkl",
#     SAVE_SEARCHLIGHTS = True
# )



# # Step 2: loading and computing the data RDMs
# if USE_PREVIOUS_DATA_RDM == True:
#     # Open the data RDM file
#     with open(f"{results_dir}/data_RDM.pkl", 'rb') as file:
#         data_RDM = pickle.load(file)
    
#     if VISUALISE_RDMS == True:
#         analyse_MRI_behav.visualise_data_RDM(mni_x = 53, 
#                                              mni_y = 30, 
#                                              mni_z = 2, 
#                                              data_RDM_file = data_RDM, 
#                                              mask = mask)
        
# else:
#     # Create dictionary to store the data for each EV for both task halves
#     EVs_both_halves_dict = {
#         '1': None,
#         '2': None
#     }
#     # create new dictionary to store the 2D array of EVs for both task halves
#     EVs_both_halves_2d = EVs_both_halves_dict.copy()

#     for split in task_halves:

#         EVs_path_dict = model_rdm_functions.get_EV_path_dict_instruction_period(
#             subject_directory = subject_directory,
#             split = split
#             )
        
#         # Load in the EVs for the instruction periods from the dictionary of paths
#         EVs_data_dict = model_rdm_functions.load_EV_data(
#             EVs_path_dict = EVs_path_dict,
#             RDM_VERSION = RDM_VERSION
#         )

#         # Convert the dictionary of EVs to a 2D array
#         EVs_data_2d = model_rdm_functions.EV_data_dict_to_2d(
#             EVs_data_dict = EVs_data_dict,
#         )
        
#         # Put the 2D array of EVs into the EV_both_halves dictionary
#         EVs_both_halves_dict[split] = EVs_data_dict
#         EVs_both_halves_2d[split] = EVs_data_2d

#         # Each part has array of shape (n_conditions, x, y, z)
#         EVs_both_halves_dict[split] = np.array(list(EVs_both_halves_dict[split].values()))

#         # Remove NaNs
#         EVs_both_halves_dict[split] = np.nan_to_num(EVs_both_halves_dict[split])

#         # Each part has array of shape (n_conditions, n_voxels)
#         EVs_both_halves_2d[split] = EVs_both_halves_dict[split].reshape(EVs_both_halves_dict[split].shape[0], -1)

#     # Combine the EVs from both task halves into a single 2D array (condition, x, y, z)
#     EVs_both_halves_array_2d = np.concatenate((EVs_both_halves_2d['1'], EVs_both_halves_2d['2']), axis=0)

#     # Remove NaNs
#     # EVs_both_halves_array_2d = np.nan_to_num(EVs_both_halves_array_2d)

#     # define the condition names for both task halves
#     # 2 * 10 conditions, since there are 10 identical executution conditions in each task half
#     data_conds = np.reshape(np.tile((np.array(['cond_%02d' % x for x in np.arange(no_RDM_conditions)])), (1, 2)).transpose(), 2 * no_RDM_conditions)

#     # Defining both task halves runs: 0s first half, 1s is second half
#     sessions = np.concatenate((np.zeros(int(EVs_both_halves_2d['1'].shape[0])),
#                                 np.ones(int(EVs_both_halves_2d['2'].shape[0]))))  
    


#     # for all other cases, cross correlated between task-halves.
#     # TODO: try to have one only half 
#     data_RDM = get_searchlight_RDMs(
#         data_2d = EVs_both_halves_2d, 
#         centers = centers, 
#         neighbors = neighbors, 
#         events  = data_conds, 
#         method  ='crosscorr', 
#         cv_descr = sessions
#                     )
    
#     # Save the data RDMs
#     with open(f"{results_dir}/data_RDM.pkl", 'wb') as file:
#         pickle.dump(data_RDM, file)


# # Step 3: Load  Model RDMs, created in `replay_RSA_model_RDM.py`
# replay_RDM_object = model_rdm_functions.load_model_RDMs(
#     RDM_dir = RDM_dir,
    
#     )

# model_RDM_dir = {}
# # Step 4: Run the RSA
# model_RDM_dir["replay"] = rsr.calc_rdm(
#     replay_RDM_object, 
#     method='correlation', 
#     descriptor='conds'
#     )


# print("Running RSA")