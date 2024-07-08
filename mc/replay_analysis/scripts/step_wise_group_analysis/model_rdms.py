"""
Alif's Functions for RSA
"""
# # Standard Libraries
import numpy as np
import pandas as pd
NaN = np.nan
import data_rdms



def create_model_rdm(conditions, TYPE='replay'):
    """
    Create a different model RDM that is based on a different hypothesis

    Parameters
        TYPE: str
        TYPEs are 
        - replay
        - replay_zero_off_diag
        - replay_nan_off_diag

    Returns
        rdm_df: pd.DataFrame


    """

    if TYPE == 'replay':

        rdm = np.array(
            [[-1,   +1,   +1/4, -1/4, +1/4, -1/4, +1/4, -1/4, +1/4, -1/4],
             [+1,   -1,   -1/4, +1/4, -1/4, +1/4, -1/4, +1/4, -1/4, +1/4],
             [+1/4, -1/4,   -1,   +1, +1/4, -1/4, +1/4, -1/4, +1/4, -1/4],
             [-1/4, +1/4,   +1,   -1, -1/4, +1/4, -1/4, +1/4, -1/4, +1/4],
             [+1/4, -1/4, +1/4, -1/4,   -1,   +1, +1/4, -1/4, +1/4, -1/4],
             [-1/4, +1/4, -1/4, +1/4,   +1,   -1, -1/4, +1/4, -1/4, +1/4],
             [+1/4, -1/4, +1/4, -1/4, +1/4, -1/4,   -1,   +1, +1/4, -1/4],
             [-1/4, +1/4, -1/4, +1/4, -1/4, +1/4,   +1,   -1, -1/4, +1/4],
             [+1/4, -1/4, +1/4, -1/4, +1/4, -1/4, +1/4, -1/4,   -1,   +1],
             [-1/4, +1/4, -1/4, +1/4, -1/4, +1/4, -1/4, +1/4,   +1,   -1]]
        )


    elif TYPE == 'replay_zero_off_diag':
        # Create a model RDM where the diagagnoal conditions are present, 
        # Everything off diagonal are zeros. This hypothesises that the other conditions are not correlated
        
        rdm = np.array(
            [[-1, +1,  0,  0,  0,  0,  0,  0,  0,   0],
             [+1, -1,  0,  0,  0,  0,  0,  0,  0,   0],
             [ 0,  0, -1, +1,  0,  0,  0,  0,  0,   0],
             [ 0,  0, +1, -1,  0,  0,  0,  0,  0,   0],
             [ 0,  0,  0,  0, -1, +1,  0,  0,  0,   0],
             [ 0,  0,  0,  0, +1, -1,  0,  0,  0,   0],
             [ 0,  0,  0,  0,  0,  0, +1,  -1, 0,   0],
             [ 0,  0,  0,  0,  0,  0, -1,  +1, 0,   0],
             [ 0,  0,  0,  0,  0,  0,  0,   0, -1, +1],
             [ 0,  0,  0,  0,  0,  0,  0,   0, +1, -1]]
        )

        
    elif TYPE == 'replay_nan_off_diag':
        # Create a model RDM where the diagagnoal conditions are present, 
        # Everything off diagonal are NaNs. This makes no assumptions about the correlations between the other pattern conditions
        rdm = np.array(
            [[-1,    +1,  NaN,  NaN,  NaN,  NaN,  NaN,  NaN,  NaN,   NaN],
             [+1,    -1,  NaN,  NaN,  NaN,  NaN,  NaN,  NaN,  NaN,   NaN],
             [ NaN, NaN,   -1,   +1,  NaN,  NaN,  NaN,  NaN,  NaN,   NaN],
             [ NaN, NaN,   +1,   -1,  NaN,  NaN,  NaN,  NaN,  NaN,   NaN],
             [ NaN, NaN,  NaN,  NaN,   -1,   +1,  NaN,  NaN,  NaN,   NaN],
             [ NaN, NaN,  NaN,  NaN,   +1,   -1,  NaN,  NaN,  NaN,   NaN],
             [ NaN, NaN,  NaN,  NaN,  NaN,  NaN,   -1,   +1,  NaN,   NaN],
             [ NaN, NaN,  NaN,  NaN,  NaN,  NaN,   +1,   -1,  NaN,   NaN],
             [ NaN, NaN,  NaN,  NaN,  NaN,  NaN,  NaN,   NaN,   -1,   +1],
             [ NaN, NaN,  NaN,  NaN,  NaN,  NaN,  NaN,   NaN,   +1,   -1]]
        )

    elif TYPE == 'difficulty':
                
        rdm = np.array(
            [[+1/4, -1/4,   +1/4, -1/4, +1/4, -1/4, +1/4, -1/4, +1/4, -1/4],
             [-1/4, +1/4,   -1/4, +1/4, -1/4, +1/4, -1/4, +1/4, -1/4, +1/4],
             [+1/4, -1/4,  +1/4, -1/4, +1/4, -1/4, +1/4, -1/4, +1/4, -1/4],
             [-1/4, +1/4,  -1/4, +1/4, -1/4, +1/4, -1/4, +1/4, -1/4, +1/4],
             [+1/4, -1/4, +1/4, -1/4,  +1/4, -1/4, +1/4, -1/4, +1/4, -1/4],
             [-1/4, +1/4, -1/4, +1/4,  -1/4, +1/4, -1/4, +1/4, -1/4, +1/4],
             [+1/4, -1/4, +1/4, -1/4, +1/4, -1/4,   +1/4, -1/4, +1/4, -1/4],
             [-1/4, +1/4, -1/4, +1/4, -1/4, +1/4,   -1/4, +1/4, -1/4, +1/4],
             [+1/4, -1/4, +1/4, -1/4, +1/4, -1/4, +1/4, -1/4,  +1/4, -1/4,],
             [-1/4, +1/4, -1/4, +1/4, -1/4, +1/4, -1/4, +1/4,  -1/4, +1/4,]]
        )

    elif TYPE == "difficulty_nan_diag":

        rdm = np.array(
            [[NaN, NaN, +1/4, -1/4, +1/4, -1/4, +1/4, -1/4, +1/4, -1/4],
             [NaN, NaN, -1/4, +1/4, -1/4, +1/4, -1/4, +1/4, -1/4, +1/4],
             [+1/4, -1/4, NaN, NaN, +1/4, -1/4, +1/4, -1/4, +1/4, -1/4],
             [-1/4, +1/4, NaN, NaN, -1/4, +1/4, -1/4, +1/4, -1/4, +1/4],
             [+1/4, -1/4, +1/4, -1/4, NaN, NaN, +1/4, -1/4, +1/4, -1/4],
             [-1/4, +1/4, -1/4, +1/4, NaN, NaN, -1/4, +1/4, -1/4, +1/4],
             [+1/4, -1/4, +1/4, -1/4, +1/4, -1/4, NaN, NaN, +1/4, -1/4],
             [-1/4, +1/4, -1/4, +1/4, -1/4, +1/4, NaN, NaN, -1/4, +1/4],
             [+1/4, -1/4, +1/4, -1/4, +1/4, -1/4, +1/4, -1/4, NaN, NaN],
             [-1/4, +1/4, -1/4, +1/4, -1/4, +1/4, -1/4, +1/4, NaN, NaN]]
        )


    rdm_df = pd.DataFrame(rdm, index=conditions[:10], columns=conditions[10:])
    return rdm_df

def get_model_rdms(
        conditions: list,
        TYPE: str =  'replay',
) -> dict:
    """
    Returns a dictionary of model RDMs for the replay analysis.
    """
    # Create the dictionary
    model_rdms_dict = {}
    # Create the model RDM for the replay analysis and store it in the dictionary
    model_rdms_dict[TYPE] = create_model_rdm(conditions, TYPE)
    # Return the dictionary
    return model_rdms_dict





# """
# Data RDM functions
# """

# def get_EV_path_dict(
#         subject_directory: str,
#         EVs_type:str = "instruction_period"
        
#     ) -> dict:
#     """
#     Function that returns the correct dictionary of EV paths that will be used to load in the correct dataset.
    
#     Param
#         subject_directory: str
#         configs: dict Structured in to two parts configs['1'] and configs['2']. Each one of these contains the execution order of the tasks
#         split: str
#         EVs_type: str

#     Returns
#         EVs_path_dict: dict

#     """
#     if EVs_type == "instruction_period":
#         # Get the path to the EVs for the instruction period
#         EVs_path_dict = get_EV_path_dict_instruction_period(subject_directory)
#     else:
#         raise ValueError(f"EVs_type {EVs_type} not found. Please use 'instruction_period'")

#     return EVs_path_dict

# def get_EV_path_dict_instruction_period(subject_directory: str) -> dict:
#     """
#     split: str is a task half ("1" or "2")
#     Returns a dictionary with the paths to the EVs for the instruction period, for a 

#     Example of the dictionary to be returned:
#     {
#         "1": {
#             ev_A1_forw_instruction_onset: "path/to/ev_A1_forw_instruction_onset.nii.gz",
#             ev_A2_forw_instruction_onset: "path/to/ev_A2_forw_instruction_onset.nii.gz",
#             ...
#         }

#         "2": {
#             ev_A1_forw_instruction_onset: "path/to/ev_A1_forw_instruction_onset.nii.gz",
#             ev_A2_forw_instruction_onset: "path/to/ev_A2_forw_instruction_onset.nii.gz",
#             ...
#         }
#     }
#     """
#     EVs_path_dict = {}
#     # pe_path is the path to of the instruction period for each of the task 

#     split = 1
#     pe_path = f"{subject_directory}/func/glm_01_pt0{split}.feat/stats"

#     with open(f"{subject_directory}/func/EVs_01_pt0{split}/task-to-EV.txt", 'r') as file:
#         for line in file:
#             index, name = line.strip().split(' ', 1)
#             EVs_path_dict[f"{name}_EV_{index}"] = os.path.join(pe_path, f"pe{int(index)+1}.nii.gz")

#     split = 2
#     pe_path = f"{subject_directory}/func/glm_01_pt0{split}.feat/stats"

#     with open(f"{subject_directory}/func/EVs_01_pt0{split}/task-to-EV.txt", 'r') as file:
#             for line in file:
#                 index, name = line.strip().split(' ', 1)
#                 EVs_path_dict[f"{name}_EV_{index}"] = os.path.join(pe_path, f"pe{int(index)+1}.nii.gz")


#     return EVs_path_dict


# def load_EV_data(EVs_path_dict: dict, RDM_VERSION) -> dict:
#     """
#     Read in dictionary of paths to EVs and load them as numpy arrays into a dictionary
#     """
#     # create data dictionary

#     # create list of tasks in the order they are in the EVs
#     EV_tasks = list(EVs_path_dict.keys())

#     # remove any tasks from EV_dict_keys that are not conditions to be compared but do have associated images
#     for task in EV_tasks:
#         if any(substring in task for substring in ['ev_press_EV_EV']):
#             EV_tasks.remove(task)

#     # Sort the order to be alphabetical
#     EV_tasks.sort()

#     EVs_data_dict = {}

#     # TODO: this EV_tasks must be in the correct order
#     # Use the same list to create the model RDm to load in the data dictinoary (to become the data RDM )
#     for task_idx, task in enumerate(EV_tasks):
#         EVs_data_dict[task] = nib.load(EVs_path_dict[task]).get_fdata()



#     return EVs_data_dict


# def EV_data_dict_to_2d(EVs_data_dict: dict) -> np.array:
#     """
#     Convert dictionary of EV data to 2D numpy array.

#     Returns 
#         The dimensions of the array are n_conditions x n_voxels
#     """
#     # Convert the list to a NumPy array
#     EVs_data_np = np.array(list(EVs_data_dict.values()))

#     # reshape data so we have n_observations x n_voxels
#     EVs_data_2d = EVs_data_np.reshape([EVs_data_np.shape[0], -1])
#     EVs_data_2d = np.nan_to_num(EVs_data_2d) # now this is 20timepoints x 746.496 voxels
    
#     return EVs_data_2d


# def get_subject_directory(data_folder):
#     """

#     """
#     data_folder = Path(data_folder)

#     return subject_directory


# def get_searchlights(
#         mask, 
#         radius = 3,
#         threshold = 0.5,
#         USE_PREVIOUS_SEARCHLIGHTS = True,
#         NEIGHBORS_PATH = None,
#         CENTERS_PATH = None,
#         SAVE_SEARCHLIGHTS = False
# ):
#     """
#     Creates the set of searchlights for the RSA analysis from the mask of th brain.
    
#     Parameters
#         mask: anatomical mask of the area of interest (the computation part of the brain)
#         radius: the radius of the searchlight
#         threshold: the threshold for the number of voxels in the searchlight that are not NaN
#     """

#     if USE_PREVIOUS_SEARCHLIGHTS == True:
#         with open(CENTERS_PATH, 'rb') as file:
#             centers = pickle.load(file)
#         with open(NEIGHBORS_PATH, 'rb') as file:
#             neighbors = pickle.load(file)


#     else:

#         # mask = mask.fget_data()
#         mask = mask.get_fdata()
#         centers, neighbors = get_volume_searchlight(mask, radius, threshold)

#         if SAVE_SEARCHLIGHTS == True:
#             with open(CENTERS_PATH, 'wb') as file:
#                 pickle.dump(centers, file)
#             with open(NEIGHBORS_PATH, 'wb') as file:
#                 pickle.dump(neighbors, file)

    

#     return centers, neighbors




# def create_data_RDMs(
 

# ):
    



#     # Step 2: loading and computing the data RDMs
#     if USE_PREVIOUS_DATA_RDM == True:
#         with open(f"{results_dir}/data_RDM.pkl", 'rb') as file:
#             data_RDM = pickle.load(file)
#         if VISUALISE_RDMS == True:
#             analyse_MRI_behav.visualise_data_RDM(mni_x=53, 
#                                                  mni_y = 30, 
#                                                  mni_z= 2, 
#                                                  data_RDM_file= data_RDM, 
#                                                  mask=mask)
            
#     else:
#         data_RDM_file_2d = {}
#         data_RDM_file = {}
#         data_RDM_file_1d = {}
#         reading_in_EVs_dict = {}
#         image_paths = {}
        
#         # I need to do this slightly differently. I want to be super careful that I create 2 'identical' splits of data.
#         # thus, check which folder has the respective task.
#         for split in sorted_keys:
#             if RDM_VERSION == '01':
#                 # DOUBLE CHECK IF THIS IS EVEN STILL CORRECT!!!
#                 # for condition 1, I am ignoring task halves. to make sure everything goes fine, use the .txt file
#                 # and only load the conditions in after the task-half loop.
#                 pe_path = f"{subject_directory}/func/glm_{REGRESSION_VERSION}_pt0{split}.feat/stats"
#                 with open(f"{subject_directory}/func/EVs_{RDM_VERSION}_pt0{split}/task-to-EV.txt", 'r') as file:
#                     for line in file:
#                         index, name = line.strip().split(' ', 1)
#                         reading_in_EVs_dict[f"{name}_EV_{index}"] = os.path.join(pe_path, f"pe{int(index)+1}.nii.gz")
#             else:           
#                 i = -1
#                 image_paths[split]   = [None] * no_RDM_conditions  # Initialize a list for each half of the dictionary
#                 data_RDM_file[split] = [None] * no_RDM_conditions  # Initialize a list for each half of the dictionary
#                 for EV_no, task in enumerate(sorted_keys[split]):
#                     for regressor_sets in reg_keys:
#                         if regressor_sets[0].startswith(task):
#                             curr_reg_keys = regressor_sets
#                     for reg_key in curr_reg_keys:
#                         # print(f"now looking for {task}")
#                         for EV_01 in reading_in_EVs_dict_01:
#                             if EV_01.startswith(reg_key):
#                                 i = i + 1
#                                 # print(f"looking for {task} and found it in 01 {EV_01}, index {i}")
#                                 image_paths[split][i] = reading_in_EVs_dict_01[EV_01]  # save path to check if everything went fine later
#                                 data_RDM_file[split][i] = nib.load(reading_in_EVs_dict_01[EV_01]).get_fdata()
#                         for EV_02 in reading_in_EVs_dict_02:
#                             if EV_02.startswith(reg_key):
#                                 i = i + 1
#                                 # print(f"looking for {task} and found it in 01 {EV_02}, index {i}")
#                                 image_paths[split][i] = reading_in_EVs_dict_02[EV_02]
#                                 data_RDM_file[split][i] = nib.load(reading_in_EVs_dict_02[EV_02]).get_fdata() 
#                                 # Convert the list to a NumPy array
                

                
#                 print(f"This is the order now: {image_paths[split]}")
#                 data_RDM_file[split] = np.array(data_RDM_file[split])
#                 # reshape data so we have n_observations x n_voxels
#                 data_RDM_file_2d[split] = data_RDM_file[split].reshape([data_RDM_file[split].shape[0], -1])
#                 data_RDM_file_2d[split] = np.nan_to_num(data_RDM_file_2d[split]) # now this is 80timepoints x 746.496 voxels
                
#                 if RDM_VERSION == f"{RDM_VERSION}_999": # scramble voxels randomly
#                     data_RDM_file_1d[split] = data_RDM_file_2d[split].flatten()
#                     np.random.shuffle(data_RDM_file_1d[split]) #shuffle all voxels randomly
#                     data_RDM_file_2d[split] = data_RDM_file_1d[split].reshape(data_RDM_file_2d[split].shape) # and reshape

        
#         if RDM_VERSION in ['01']:
#             # "data_RDM_file" is a dictionary with a key for each RDM_version
#             data_RDM_file_2d = {}
#             data_RDM_file = {}
#             data_RDM_file[RDM_VERSION] = [None] * no_RDM_conditions

#             # create list of tasks in the order they are in the EVs
#             EV_dict_keys = list(reading_in_EVs_dict.keys())

#             for task in EV_dict_keys:
#                 # remove any tasks from EV_dict_keys that are not conditions that have associated images
#                 if any(substring in task for substring in ['ev_press_EV_EV']):
#                     EV_dict_keys.remove(task)

#             # For all remaining conditions, load the images and store them in the data_RDM_file dictionary
#             for task_idx, task in enumerate(EV_dict_keys):
#                 image_paths[task_idx] = reading_in_EVs_dict[task]
#                 data_RDM_file[RDM_VERSION][task_idx] = nib.load(image_paths[task_idx]).get_fdata()
#             # Convert the list to a NumPy array
#             data_RDM_file_np = np.array(data_RDM_file[RDM_VERSION])
#             # reshape data so we have n_observations x n_voxels
#             data_RDM_file_2d = data_RDM_file_np.reshape([data_RDM_file_np.shape[0], -1])
#             data_RDM_file_2d = np.nan_to_num(data_RDM_file_2d) # now this is 20timepoints x 746.496 voxels

#             # image_paths is a dictionary, containing the paths for the images (3D arrays) for each of the EVs being tested
#             print(f"This is the order now: {image_paths}")  



#         # define the conditions, combine both task halves
#         data_conds = np.reshape(np.tile((np.array(['cond_%02d' % x for x in np.arange(no_RDM_conditions)])), (1,2)).transpose(),2*no_RDM_conditions)  
#         # now prepare the data RDM file. 
#         # final data RDM file; 
#         if RDM_VERSION in ['01', '01-1', '01-2']:
#             data_conds = np.reshape(np.tile((np.array(['cond_%02d' % x for x in np.arange(no_RDM_conditions)])), (1)).transpose(),no_RDM_conditions)  
#             data_RDM = get_searchlight_RDMs(data_RDM_file_2d, centers, neighbors, data_conds, method='correlation')
#         else:
#             # this is defining both task halves/ runs: 0 is first half, the second one is 1s
#             sessions = np.concatenate((np.zeros(int(data_RDM_file['1'].shape[0])), np.ones(int(data_RDM_file['2'].shape[0]))))  
#             # for all other cases, cross correlated between task-halves.
#             data_RDM = get_searchlight_RDMs(data_RDM_file_2d, centers, neighbors, data_conds, method='crosscorr', cv_descr=sessions)
#             # save  so that I don't need to recompute - or don't save bc it's massive
#             # with open(f"{results_dir}/data_RDM.pkl", 'wb') as file:
#                 # pickle.dump(data_RDM, file)

#     return data_RDM


# def create_combination_model():
#     """
#     # Step 5: Create Combination models
#     # I am interested in:
#     # combo clocks with midnight, phase, state and location included
#     # combo split clocks with now, one future, two future, [three future]
#     """    

#     if RDM_VERSION in ['01']:
#         instruction_comp_RDM = rsatoolbox.rdm.concat(model_RDM_dir['direction_presentation'], model_RDM_dir['execution_similarity'], model_RDM_dir['presentation_similarity'])
#         instruction_comp_model = rsatoolbox.model.ModelWeighted('instruction_comp_RDM', instruction_comp_RDM)
#         results_instruction_comp_model = Parallel(n_jobs=3)(delayed(analyse_MRI_behav.evaluate_model)(instruction_comp_model, d) for d in tqdm(data_RDM, desc='running GLM for all searchlights in combo model - different instruction models'))
        

#         file_names = ['DIRECTION-PRESENTATION-combo-instr', 
#                       'EXECUTION-SIM-combo-instr', 
#                       'PRESENTATION-SIM-combo-instr']
        
#         for number_regr, file_name in enumerate(file_names):
#             # save the results
#             analyse_MRI_behav.save_RSA_result(
#                 result_file= results_instruction_comp_model, 
#                 data_RDM_file= data_RDM, 
#                 file_path = results_dir, 
#                 file_name= file_name, 
#                 mask= mask, 
#                 number_regr = number_regr, 
#                 ref_image_for_affine_path=ref_img
#                 )

#      # combo clocks and controls    
#     if RDM_VERSION == '02': # modelling all
#         # first: clocks with midnight, phase, state and location.
#         clocks_midn_states_loc_ph_RDM = rsatoolbox.rdm.concat(model_RDM_dir['clocks'], model_RDM_dir['midnight'], model_RDM_dir['state'], model_RDM_dir['location'], model_RDM_dir['phase'])
#         clocks_midn_states_loc_ph_model = rsatoolbox.model.ModelWeighted('clocks_midn_states_RDM', clocks_midn_states_loc_ph_RDM)
#         # the first is t, the second beta. [est.tvalues[1:], est.params[1:]]
#         results_clocks_midn_states_loc_ph_model = Parallel(n_jobs=3)(delayed(analyse_MRI_behav.evaluate_model)(clocks_midn_states_loc_ph_model, d) for d in tqdm(data_RDM, desc='running GLM for all searchlights in combo model - clocks vs. phase, midn, state, loc'))
        
#         file_names = ['CLOCK-combo-cl-mid-st-loc-ph', 
#                       'MIDN-combo_cl-mid-st-loc-ph', 
#                       'STATE-combo_cl-mid-st-loc-ph', 
#                       'LOC-combo_cl-mid-st-loc-ph', 
#                       'PHASE-combo_cl-mid-st-loc-ph']
        
#         for number_regr, file_name in enumerate(file_names):
#             analyse_MRI_behav.save_RSA_result(
#                 result_file=results_clocks_midn_states_loc_ph_model, 
#                 data_RDM_file=data_RDM, 
#                 file_path = results_dir, 
#                 file_name= file_name, 
#                 mask=mask, 
#                 number_regr = number_regr, 
#                 ref_image_for_affine_path=ref_img
#                 )


#      # combo clocks and controls
#     if RDM_VERSION in ['02', '02-A'] and REGRESSION_VERSION in ['03', '03-4', '04', '04-4']: # don't model location and midnight together if reduced to reward times as they are the same.
#         # first: clocks with midnight, phase, state and location.
#         clocks_midn_states_ph_RDM = rsatoolbox.rdm.concat(model_RDM_dir['clocks'], model_RDM_dir['midnight'], model_RDM_dir['state'], model_RDM_dir['phase'])
#         clocks_midn_states_ph_model = rsatoolbox.model.ModelWeighted('clocks_midn_states_RDM', clocks_midn_states_ph_RDM)
#         # the first is t, the second beta. [est.tvalues[1:], est.params[1:]]
#         results_clocks_midn_states_ph_model = Parallel(n_jobs=3)(delayed(analyse_MRI_behav.evaluate_model)(clocks_midn_states_ph_model, d) for d in tqdm(data_RDM, desc='running GLM for all searchlights in combo model - clocks vs. phase, midn, state')) 

#         file_names = ['CLOCK-combo-cl-mid-st-ph', 
#                       'MIDN-combo_cl-mid-st-ph', 
#                       'STATE-combo_cl-mid-st-ph', 
#                       'PHASE-combo_cl-mid-st-ph']
        
#         for number_regr, file_name in enumerate(file_names):
#             analyse_MRI_behav.save_RSA_result(
#                 result_file=results_clocks_midn_states_ph_model, 
#                 data_RDM_file=data_RDM, 
#                 file_path = results_dir, 
#                 file_name= file_name, 
#                 mask=mask, 
#                 number_regr = number_regr, 
#                 ref_image_for_affine_path=ref_img
#                 )

#     # combo clocks and controls
#     if RDM_VERSION == '03' and REGRESSION_VERSION in ['02', '04']: # modeling only reward rings
#         # first: clocks with midnight, phase, state and location.
#         clocks_midn_states_loc_ph_RDM = rsatoolbox.rdm.concat(model_RDM_dir['clocks_only-rew'], model_RDM_dir['midnight_only-rew'], model_RDM_dir['state'], model_RDM_dir['location'], model_RDM_dir['phase'])
#         clocks_midn_states_loc_ph_model = rsatoolbox.model.ModelWeighted('clocks_midn_states_loc_ph_RDM', clocks_midn_states_loc_ph_RDM)
#         # the first is t, the second beta. [est.tvalues[1:], est.params[1:]]
#         results_clocks_midn_states_loc_ph_model = Parallel(n_jobs=3)(delayed(analyse_MRI_behav.evaluate_model)(clocks_midn_states_loc_ph_model, d) for d in tqdm(data_RDM, desc='running GLM for all searchlights in combo model - clocks vs. phase, midn, state, loc'))
        
#         file_names = ['CLOCKrw-combo-cl-mid-st-loc-ph', 
#                       'MIDNrw-combo_cl-mid-st-loc-ph', 
#                       'STATE-combo_cl-mid-st-loc-ph', 
#                       'LOC-combo_cl-mid-st-loc-ph', 
#                       'PHASE-combo_cl-mid-st-loc-ph']
        
#         for number_regr, file_name in enumerate(file_names):
#             analyse_MRI_behav.save_RSA_result(
#                 result_file=results_clocks_midn_states_loc_ph_model, 
#                 data_RDM_file=data_RDM, 
#                 file_path = results_dir, 
#                 file_name= file_name, mask=mask, 
#                 number_regr = number_regr, 
#                 ref_image_for_affine_path=ref_img)


#      # combo clocks and controls
#     if RDM_VERSION == '03' and REGRESSION_VERSION in ['03']: # don't model location and midnight together if reduced to reward times as they are the same.
#         # first: clocks with midnight, phase, state and location.
#         clocks_midn_states_loc_ph_RDM = rsatoolbox.rdm.concat(model_RDM_dir['clocks_only-rew'], model_RDM_dir['midnight_only-rew'], model_RDM_dir['state'], model_RDM_dir['phase'])
#         clocks_midn_states_loc_ph_model = rsatoolbox.model.ModelWeighted('clocks_midn_states_loc_ph_RDM', clocks_midn_states_loc_ph_RDM)
#         # the first is t, the second beta. [est.tvalues[1:], est.params[1:]]
#         results_clocks_midn_states_loc_ph_model = Parallel(n_jobs=3)(delayed(analyse_MRI_behav.evaluate_model)(clocks_midn_states_loc_ph_model, d) for d in tqdm(data_RDM, desc='running GLM for all searchlights in combo model - clocks vs. phase, midn, state, loc'))
        
#         file_names = ['CLOCKrw-combo-cl-mid-st-ph', 
#                       'MIDNrw-combo_cl-mid-st-ph', 
#                       'STATE-combo_cl-mid-st-ph', 
#                       'PHASE-combo_cl-mid-st-ph']
        
#         for number_regr, file_name in enumerate(file_names):

#             analyse_MRI_behav.save_RSA_result(
#                 result_file=results_clocks_midn_states_loc_ph_model, 
#                 data_RDM_file=data_RDM, 
#                 file_path = results_dir, 
#                 file_name= file_name, 
#                 mask=mask, 
#                 number_regr = number_regr, 
#                 ref_image_for_affine_path=ref_img)

#      # combo clocks and controls
#     if RDM_VERSION == '03-1' and REGRESSION_VERSION in ['03', '03-4', '03-l', '03-e']:
#         added_rew_locs_loc_ph_st_RDM = rsatoolbox.rdm.concat(model_RDM_dir['curr-and-future-rew-locs'], model_RDM_dir['location'], model_RDM_dir['phase'], model_RDM_dir['state'])
#         added_rew_locs_loc_ph_st_model = rsatoolbox.model.ModelWeighted('added_rew_locs_loc_ph_st_RDM', added_rew_locs_loc_ph_st_RDM)
#         results_added_rew_locs_loc_ph_st_model = Parallel(n_jobs=3)(delayed(analyse_MRI_behav.evaluate_model)(added_rew_locs_loc_ph_st_model, d) for d in tqdm(data_RDM, desc='running GLM for all searchlights in combo model - current/fut rew locs clock vs. phase, state, loc'))
        
#         file_names = ['CLOCKrewloc-combo-clrw-loc-ph-st',
#                         'LOC-combo-clrw-loc-ph-st',
#                         'PHASE-combo-clrw-loc-ph-st',
#                         'STATE-combo-clrw-loc-ph-st']
        
#         for number_regr, file_name in enumerate(file_names):
#             analyse_MRI_behav.save_RSA_result(
#                 result_file=results_added_rew_locs_loc_ph_st_model, 
#                 data_RDM_file=data_RDM, 
#                 file_path = results_dir, 
#                 file_name= file_name, 
#                 mask=mask, 
#                 number_regr = number_regr, 
#                 ref_image_for_affine_path=ref_img
#                 )

#     # combo clocks and controls
#     if RDM_VERSION == '04': #modelling only path rings  
#         # first: clocks with midnight, phase, state and location.
#         clocks_midn_states_loc_ph_RDM = rsatoolbox.rdm.concat(model_RDM_dir['clocks_no-rew'], model_RDM_dir['midnight_no-rew'], model_RDM_dir['state'], model_RDM_dir['location'], model_RDM_dir['phase'])
#         clocks_midn_states_loc_ph_model = rsatoolbox.model.ModelWeighted('clocks_midn_states_loc_ph_RDM', clocks_midn_states_loc_ph_RDM)
#         # the first is t, the second beta. [est.tvalues[1:], est.params[1:]]
#         results_clocks_midn_states_loc_ph_model = Parallel(n_jobs=3)(delayed(analyse_MRI_behav.evaluate_model)(clocks_midn_states_loc_ph_model, d) for d in tqdm(data_RDM, desc='running GLM for all searchlights in combo model - clocks vs. phase, midn, state, loc'))
        
#         file_names = ['CLOCKnorw-combo-cl-mid-st-loc-ph',
#                       'MIDNnorw-combo_cl-mid-st-loc-ph',
#                       'STATE-combo_cl-mid-st-loc-ph',
#                       'LOC-combo_cl-mid-st-loc-ph',
#                       'PHASE-combo_cl-mid-st-loc-ph']

#         for number_regr, file_name in enumerate(file_names):
#             analyse_MRI_behav.save_RSA_result(
#                 result_file=results_clocks_midn_states_loc_ph_model, 
#                 data_RDM_file=data_RDM, 
#                 file_path = results_dir, 
#                 file_name= file_name, 
#                 mask=mask, 
#                 number_regr = number_regr, 
#                 ref_image_for_affine_path=ref_img
#                 )

#     # combo split clocks
#     if RDM_VERSION in ['02', '03', '04', '02-A']:
#         # second: split clock: now/ midnight; one future, two future, three future
#         split_clocks_RDM = rsatoolbox.rdm.concat(model_RDM_dir['curr_rings_split_clock'], model_RDM_dir['one_fut_rings_split_clock'], model_RDM_dir['two_fut_rings_split_clock'], model_RDM_dir['three_fut_rings_split_clock'])
#         split_clocks_model = rsatoolbox.model.ModelWeighted('split_clocks_RDM', split_clocks_RDM)
#         results_split_clocks_combo_model = Parallel(n_jobs=3)(delayed(analyse_MRI_behav.evaluate_model)(split_clocks_model, d) for d in tqdm(data_RDM, desc='running GLM for all searchlights in combo model - split clocks'))
        
#         file_names = ['CURR-RINGS_combo_split_clock',
#                       'ONE-FUTR-RINGS_combo_split_clock',
#                       'TWO-FUTR-RINGS_combo_split_clock',
#                       'THRE-FUTR-RINGS_combo_split_clock']
        
#         for number_regr, file_name in enumerate(file_names):
#             analyse_MRI_behav.save_RSA_result(
#                 result_file=results_split_clocks_combo_model, 
#                 data_RDM_file=data_RDM, 
#                 file_path = results_dir, 
#                 file_name= file_name, 
#                 mask=mask, 
#                 number_regr = number_regr, 
#                 ref_image_for_affine_path=ref_img
#                 )
            

#     # combo split clocks    
#     if RDM_VERSION in ['03-1'] and REGRESSION_VERSION in ['03', '03-4','03-l', '03-e']:
#         split_clocks_RDM = rsatoolbox.rdm.concat(model_RDM_dir['location'], model_RDM_dir['one_future_rew_loc'], model_RDM_dir['two_future_rew_loc'], model_RDM_dir['three_future_rew_loc'])
#         split_clocks_model = rsatoolbox.model.ModelWeighted('split_clocks_RDM', split_clocks_RDM)
        
#         results_current_and_all_future_rew_model = Parallel(n_jobs=3)(delayed(analyse_MRI_behav.evaluate_model)(split_clocks_model, d) for d in tqdm(data_RDM, desc='running GLM for all searchlights in combo model - split clocks after regression'))
        
#         file_names = ['CURR_REW-combo_split-clock',
#                       'ONE-FUT-REW_combo_split-clock',
#                       'TWO-FUT-REW_combo_split-clock',
#                       'THREE-FUT-REW_combo_split-clock']
        
#         for number_regr, file_name in enumerate(file_names):
#             analyse_MRI_behav.save_RSA_result(
#                 result_file=results_current_and_all_future_rew_model, 
#                 data_RDM_file=data_RDM, 
#                 file_path = results_dir, 
#                 file_name= file_name, 
#                 mask=mask, 
#                 number_regr = number_regr, 
#                 ref_image_for_affine_path=ref_img
#                 )
            
#     # combo split clocks with state to control
#     if RDM_VERSION in ['03-1'] and REGRESSION_VERSION in ['03', '03-4','03-l', '03-e']:
#         split_clocks_state_RDM = rsatoolbox.rdm.concat(model_RDM_dir['location'], model_RDM_dir['one_future_rew_loc'], model_RDM_dir['two_future_rew_loc'], model_RDM_dir['three_future_rew_loc'], model_RDM_dir['state'])
#         split_clocks_state_model = rsatoolbox.model.ModelWeighted('split_clocks_RDM', split_clocks_state_RDM)
        
#         results_current_and_all_future_rew_state_model = Parallel(n_jobs=3)(delayed(analyse_MRI_behav.evaluate_model)(split_clocks_state_model, d) for d in tqdm(data_RDM, desc='running GLM for all searchlights in combo model - split clocks after regression plus state'))
        
#         file_names = ['CURR_REW-combo_split-clock-state',
#                       'ONE-FUT-REW_combo_split-clock-state',
#                       'TWO-FUT-REW_combo_split-clock-state',
#                       'THREE-FUT-REW_combo_split-clock-state',
#                       'STATE_combo_split-clock-state']
        
#         for number_regr, file_name in enumerate(file_names):
#             analyse_MRI_behav.save_RSA_result(
#                 result_file=results_current_and_all_future_rew_state_model, 
#                 data_RDM_file=data_RDM, 
#                 file_path = results_dir, 
#                 file_name= file_name, 
#                 mask=mask, 
#                 number_regr = number_regr, 
#                 ref_image_for_affine_path=ref_img
#                 )
#     pass




# """
# Model RSA functions
# """



# """
# Different functions for different models
# """


# def create_matrix_replay(EVs):
#     """

#     """
#     # create the empty RDM array
#     rdm_array = np.zeros((len(EVs), len(EVs)))

#     print(EVs[0][0], EVs[0][1], EVs[0][3])
#     print(EVs[3][0], EVs[3][1], EVs[3][3])

#     # For first condition
#     for EV1_idx, EV1 in enumerate(EVs):
#         # for second condition
#         for EV2_idx, EV2 in enumerate(EVs):
#             # compare the conditions
#             # print(EV1, EV2)
#             if EV1[0] == EV2[0]: #(A, B, C, D, E) (A == A)
#                 # the set of rewards are the same

#                 if EV1[1] == EV2[1]: # Task half (1 or 2) (A1 == A1)
#                     # Forwards v.s. backwards

#                     if EV1[3] == EV2[3]: # Forwards v.s. backwards 
#                         # A1F v.s. A1F are the same so will be -1 dissimilarity
#                         # print("Same task, task half and direction")
#                         rdm_array[EV1_idx, EV2_idx] = -1

#                     else:
#                         # A1F v.s. A1B are different so will be +1 dissimilarity
#                         rdm_array[EV1_idx, EV2_idx] = +1


#                 else :# (A1 vs A2)

#                     if EV1[3] == EV2[3]: #(A1F, A2F)
#                         rdm_array[EV1_idx, EV2_idx] = -1
            
#                     else: # (A1F, A2B)
#                         rdm_array[EV1_idx, EV2_idx] = +1
            
#             else: #(A, B)

#                 if EV1[1] == EV2[1]: # (part 1 or 2)

#                     if EV1[3] == EV2[3]: # (forwards or backwards)
#                         # A1F v.s. B1F
#                         rdm_array[EV1_idx, EV2_idx] = +1/4
                    
#                     else: # (A1F, B1B)
#                         rdm_array[EV1_idx, EV2_idx] = -1/4

#                 else:
                    
#                     if EV1[3] == EV2[3]:
#                         rdm_array[EV1_idx, EV2_idx] = +1/4
#                     else:
#                         rdm_array[EV1_idx, EV2_idx] = -1/4

#     return rdm_array


# def crosscorr_matrix(
#     EVs: list
# ):
#     """
#     Create Cross-correlation matrix for the replay model

#     Parameters
#         EVs: list

#     Returns
#         matrix: np.array
#     """
#     rdm_array = create_matrix_replay(EVs)

#     # Get the bottom left corner of the matrix (10x10)
#     rdm_array  = rdm_array[10:20, 0:10]



#     # Create the cross-correlation matrix
#     # matrix = np.array(
#     #     [
#     #         [-1  , +1  , +1/4, -1/4, +1/4, -1/4, +1/4, -1/4,  +1/4, -1/4],
#     #         [+1  , -1  , +1/4, -1/4, +1/4, -1/4, +1/4, -1/4,  +1/4, -1/4],
#     #         [+1/4, -1/4, -1  , +1  , +1/4, -1/4, +1/4, -1/4,  +1/4, -1/4],
#     #         [-1/4, +1/4, +1  , -1  , +1/4, -1/4, +1/4, -1/4,  +1/4, -1/4],
#     #         [+1/4, -1/4, +1/4, -1/4, -1  , +1  , +1/4, -1/4,  +1/4, -1/4],
#     #         [-1/4, +1/4, -1/4, +1/4, +1  , -1  , +1/4, -1/4,  +1/4, -1/4],
#     #         [+1/4, -1/4, +1/4, -1/4, +1/4, -1/4, -1  , +1  ,  +1/4, -1/4],
#     #         [-1/4, +1/4, -1/4, +1/4, -1/4, +1/4, +1  , -1  ,  +1/4, -1/4],
#     #         [+1/4, -1/4, +1/4, -1/4, +1/4, -1/4, +1/4, -1/4,   -1 , +1  ],
#     #         [-1/4, +1/4, -1/4, +1/4, -1/4, +1/4, -1/4, +1/4,   +1 , -1  ]
#     #      ]
#     # )

#     assert rdm_array.shape == (10, 10), "Matrix is not the correct shape. It should be 10x10."

#     return rdm_array


# """
# Function for making the model RDM from the functions above in the correct format

# """

# def task_similarity_matrix(
#         configs_dict: dict,
#         model: list = "replay",
#         RDM_dir: str = None,
#         VISUALISE: bool = False
# ) -> RDMs_object:
#     """
#     Param
#         configs_dict: dict

#     Returns
#         model_RDM_dict: rsa.RDMs object for one model

#     """
    
#     # Get the order of the tasks from the configs dictionary
#     # sorted_keys_dict = extract_and_clean.order_task_according_to_rewards(configs_dict)

#     # list of conditions for both task halves in the order they are loaded into the data RDM for all searchlights function
#     EVs = list(configs_dict['1']) + list(configs_dict['2'])
    
#     if model == "replay":
#         # Create the RDMs for the replay analysis
#         replay_RDM = create_matrix_replay(EVs)
#         rdm_stack = np.stack([replay_RDM])
#     elif model == "replay-2":
#         # Please note that the way this matrix is made requires that all of the first half of the task are the first 10 EVs and the second are the last 10 EVs
#         replay_RDM = crosscorr_matrix(EVs)

#         rdm_stack = replay_RDM.reshape(1, 10, 10)

#         replay_RDM_vector = replay_RDM[np.triu_indices(replay_RDM.shape[0])]
#         replay_RDM_vector = replay_RDM_vector.reshape(1, -1)

        
#     else: 
#         # There is space here to add more models with new functions
#         raise ValueError(f"Model not found. The {model} model needs to be implemented")

#     # stack of RDMs for each condition. in this case, only one RDM. the "replay" one

#     # a dictonary containing all the model desciptors
#     model_RDM_descripter = {} 
#     model_RDM_descripter['replay'] = "Replay Model"

#     # Create rdm_descriptor dictionary
#     # rdm_descriptor = {}
#     # for idx, EV in enumerate(EVs):
#     #     rdm_descriptor[f"condition_{idx}"] = EV

#     # A dictionary containing the conditions' title for each EV that is being compared.
#     # pattern_descriptors = {}
#     # pattern_descriptors['conditions'] = [EV for EV in EVs]

#     # Create the RDM object
#     replay_RDM_object = RDMs_object(
#         dissimilarities       =  np.random.rand(rdm_stack.shape[0], rdm_stack.shape[1], rdm_stack.shape[2]),
#         dissimilarity_measure = 'Arbitrary',
#         descriptors           = model_RDM_descripter,
#         # rdm_descriptors       = rdm_descriptor,
#         # pattern_descriptors   = pattern_descriptors
#     )

#     # Add the dissimilarities vector to the RDM object that is correct, after the RDM object has been created
#     replay_RDM_object.dissimilarities = replay_RDM_vector

#     # Save the RDMs object to a pickle file if a directory is present
#     if RDM_dir is not None:
#         # Save the RDMs object to a pickle file
#         with open(f"{RDM_dir}/replay_RDM_object.pkl", 'wb') as file:
#             pickle.dump(replay_RDM_object, file)
#         # replay_RDM_object.save(
#         #     filename = f"{RDM_dir}/replay_RDM_object.pkl",
#         #     overwrite = True)

#     # Visualise the RDMs
#     if VISUALISE == True:

#         # Visualise the RDMs
#         v.plot_RDM_object(replay_RDM_object, 
#                           title = list(model_RDM_descripter.values())[0],
#                         #   conditions = 
#                         #   conditions = replay_RDM_object.pattern_descriptors['conditions']
#                           )

#     # return RDM_object
#     return replay_RDM_object



# def load_model_RDMs(
#         RDM_dir: str,
#     ):
#     """
#     Load in the model_RDMs object from a pickle file

#     Returns
#         Model RDMs object from rsa.rdm.rdms.RDMs

#     """
#     with open(f"{RDM_dir}/replay_RDM_object.pkl", 'rb') as file:
#         replay_RDM_object = pickle.load(file)

#     return replay_RDM_object




# def get_model_RDM(
#         configs_dict: dict,
#         USE_NEURON_MODEL = False, 
#     ) -> dict: 
#     """
#     Param
#         USE_NEURON_MODEL: bool. Decides if the neuron (vector) model is used or if the RDM is created bespoke

#     Returns
#         rdm: rsa.rdm.rdms.RDMs Class with the correct RDM within in
#     """

#     configs_order = {
#         "1": configs_dict["1"].keys(),
#         "2": configs_dict["2"].keys()
#     }

    
    
#     # rdm = RDMs(
#     #     dissimilarities = RDM
#     # )

#     # return rdm


# def 


# def create_matrix_replay(EVs):
#     """
#     Returns the RDM for the replay model
#     """
#     # create the empty RDM array
#     rdm_array = np.zeros((len(EVs), len(EVs)))

#     # For first condition
#     for EV1_idx, EV1 in enumerate(EVs):
#         # for second condition
#         for EV2_idx, EV2 in enumerate(EVs):
#             # compare the conditions
            
#             if EV1[0] == EV2[0]: #(A, B, C, D, E) (A == A)
#                 # the set of rewards are the same

#                 if EV1[1] == EV2[1]: # Task half (1 or 2) (A1 == A1)
#                     # Forwards v.s. backwards

#                     if EV1[3] == EV2[3]: # Forwards v.s. backwards 
#                         # 
#                         pass

#                     else:
#                         pass
            
#                 else :# (A1, A2)
#                     if EV1[3] == EV2[3]: #(A1F, A2F)
#                         rdm_array[EV1_idx, EV2_idx] = -1
#                     else: # (A1F, A2B)
#                         rdm_array[EV1_idx, EV2_idx] = +1
#             else: #(A, B)

#                 if EV1[1] == EV2[1]: # A

#                     if EV1[3] == EV2[3]: # (A1F, B1F)
#                         rdm_array[EV1_idx, EV2_idx] = +1/4
                    
#                     else: # (A1F, B1B)
#                         rdm_array[EV1_idx, EV2_idx] = -1/4

#                 else:
#                     pass


#     return rdm_array


# """
# Function for making the model RDM from the functions above in the correct format

# """

# def task_similarity_matrix(
#         configs_dict: dict,
#         model: list = "replay",
#         RDM_dir: str = None,
#         VISUALISE: bool = False
# ):
#     """
#     Param
#         configs_dict: dict

#     Returns
#         model_RDM_dict: rsa.RDMs object for one model

#     """
    
#     # Get the order of the tasks from the configs dictionary
#     sorted_keys_dict = extract_and_clean.order_task_according_to_rewards(configs_dict)

#     # list of conditions
#     EVs = list(sorted_keys_dict['1']) + list(sorted_keys_dict['2'])
    
#     if model == "replay":
#         # Create the RDMs for the replay analysis
#         replay_RDM = create_matrix_replay(EVs)
#     else: 
#         # There is space here to add more models with new functions
#         raise ValueError("Model not found.")

#     # stack of RDMs for each condition. in this case, only one RDM. the "replay" one
#     rdm_stack = np.stack([replay_RDM])

#     # a dictonary containing all the model desciptors
#     model_RDM_descripter = {} 
#     model_RDM_descripter['replay'] = "Replay Model"

#     # Create the RDM object
#     replay_RDM_object = RDMs_object(
#         dissimilarities = rdm_stack,
#         dissimilarity_measure = 'Arbitrary',
#         descriptors = model_RDM_descripter,
#         # pattern_descriptors =
#     )

#     if RDM_dir is not None:
#         # Save the RDMs object to a pickle file
#         with open(f"{RDM_dir}/replay_RDM_object.pkl", 'wb') as file:
#             pickle.dump(replay_RDM_object, file)

#     if VISUALISE == True:

#         # Visualise the RDMs
#         v.plot_RDM_object(replay_RDM_object, 
#                           title = list(model_RDM_descripter.values())[0],
#                           conditions = EVs,)

#     return replay_RDM_object



# def load_model_RDMs(
#         RDM_dir: str,
#     ):
#     """
#     Load in the model_RDMs object from a pickle file
#     """
#     with open(f"{RDM_dir}/replay_RDM_object.pkl", 'rb') as file:
#         replay_RDM_object = pickle.load(file)

#     return replay_RDM_object


if __name__ == '__main__':
    pass