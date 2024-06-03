"""
Functions for creating the data RDMs from the searchlights
"""
import numpy as np
import pandas as pd
import nilearn
import nibabel as nib
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import statsmodels.api  as sm
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist, squareform

def get_EV_path_dict(
        subject_directory: str,
        EVs_type:str = "instruction_period"
        
    ) -> dict:
    """
    Function that returns the correct dictionary of EV paths that will be used to load in the correct dataset.
    
    Param
        subject_directory: str
        configs: dict Structured in to two parts configs['1'] and configs['2']. Each one of these contains the execution order of the tasks
        split: str
        EVs_type: str

    Returns
        EVs_path_dict: dict

    """
    if EVs_type == "instruction_period":
        # Get the path to the EVs for the instruction period
        EVs_path_dict = get_EV_path_dict_instruction_period(subject_directory)
    else:
        raise ValueError(f"EVs_type {EVs_type} not found. Please use 'instruction_period'")

    return EVs_path_dict

def get_EV_path_dict_instruction_period(subject_directory: str) -> dict:
    """
    split: str is a task half ("1" or "2")
    Returns a dictionary with the paths to the EVs for the instruction period, for a 

    Example of the dictionary to be returned:
    {
        "1": {
            ev_A1_forw_instruction_onset: "path/to/ev_A1_forw_instruction_onset.nii.gz",
            ev_A2_forw_instruction_onset: "path/to/ev_A2_forw_instruction_onset.nii.gz",
            ...
        }

        "2": {
            ev_A1_forw_instruction_onset: "path/to/ev_A1_forw_instruction_onset.nii.gz",
            ev_A2_forw_instruction_onset: "path/to/ev_A2_forw_instruction_onset.nii.gz",
            ...
        }
    }
    """
    EVs_path_dict = {}
    # pe_path is the path to of the instruction period for each of the task 

    split = 1
    pe_path = f"{subject_directory}/func/glm_01_pt0{split}.feat/stats"

    with open(f"{subject_directory}/func/EVs_01_pt0{split}/task-to-EV.txt", 'r') as file:
        for line in file:
            index, name = line.strip().split(' ', 1)
            EVs_path_dict[f"{name}_EV_{index}"] = os.path.join(pe_path, f"pe{int(index)+1}.nii.gz")

    split = 2
    pe_path = f"{subject_directory}/func/glm_01_pt0{split}.feat/stats"

    with open(f"{subject_directory}/func/EVs_01_pt0{split}/task-to-EV.txt", 'r') as file:
            for line in file:
                index, name = line.strip().split(' ', 1)
                EVs_path_dict[f"{name}_EV_{index}"] = os.path.join(pe_path, f"pe{int(index)+1}.nii.gz")


    return EVs_path_dict




def load_EV_data(EVs_path_dict: dict, RDM_VERSION) -> dict:
    """
    Read in dictionary of paths to EVs and load them as numpy arrays into a dictionary
    """
    # create data dictionary

    # create list of tasks in the order they are in the EVs
    EV_tasks = list(EVs_path_dict.keys())

    # remove any tasks from EV_dict_keys that are not conditions to be compared but do have associated images
    for task in EV_tasks:
        if any(substring in task for substring in ['ev_press_EV_EV']):
            EV_tasks.remove(task)

    # Sort the order to be alphabetical
    EV_tasks.sort()

    EVs_data_dict = {}

    # TODO: this EV_tasks must be in the correct order
    # Use the same list to create the model RDm to load in the data dictinoary (to become the data RDM )
    for task_idx, task in enumerate(EV_tasks):
        EVs_data_dict[task] = nib.load(EVs_path_dict[task]).get_fdata()



    return EVs_data_dict


def unravel_EV_data(
        EV_data_dict: dict,
):
    """
    Unravels the data in the EV_data_dict dictionary to be a 1D array.

    Args:
        EV_data_dict (dict): A dictionary containing the EV data.

    Returns:
        dict: The updated dictionary with the data unraveled to a 1D array.
    """
    for key in EV_data_dict:
        # Unravel the data to be a 1D array. This allows conversion to pd.DataFrame
        EV_data_dict[key] = EV_data_dict[key].ravel()

        # keep a subset of the data for testing
        EV_data_dict[key] = EV_data_dict[key]

    return EV_data_dict


def resize_neighbors(vol_neighbors, size):
    """
    Resize the volume neighbours to be the correct size. 
    It will fill the array with 0s if the array is too small

    Param
    - vol_neighbours: list of arrays. Each array is the volume neighbours for a center voxel
    - size: int. The size of the volume neighbours array

    Returns
    - vol_neighbours: list of arrays. As above but with the correct size
    """
    for vol in vol_neighbors:
        if len(vol) != size:
            vol = vol.resize(size, refcheck=False)
    return vol_neighbors

def get_data_searchlight(
        vol_searchlight: pd.DataFrame,
        EVs_data_dict: pd.DataFrame
    ) -> dict:
    """
    From the vol_searchlights that are generated from the masks, we can convert them to the data searchlights
    They the table has the same formatting as the vol_searchlight table, but the values are the EVs data (bold signal for the conditions)
    The data_searchlight table will be used to calculate the RDMs for each searchlight

    Gets the data from each searchlight volume and returns the data

    NOTE: This function is super inefficient for some reason

    Param
    - vol_searchlight: pd.DataFrame. column titles are the voxel centres columns are  the volume neighbours
    - EV_data_dict: pd.DataFrame. column titles are the EVs (condiitons) and the rows are the values of the bold signal 
    
    """
    # return a df with the column headings as the centres, and the rows be the bold values 
    data_searchlights = {}

    with tqdm(total=len(vol_searchlight), desc='Creating searchlights...') as pbar:
        # def process_center(center):
        #     searchlight_indices = vol_searchlight[center]
        #     # seachlight should be a 1D array of the voxels (small array of 93 voxels)
        #     searchlight = {}
        #     for condition in EVs_data_dict:
        #         data_condition_searchlight = EVs_data_dict[condition][searchlight_indices]
                
        #         searchlight[condition] = data_condition_searchlight
            
        #     # convert the 93 x 20 array to a dataframe
        #     searchlight = pd.DataFrame(searchlight)
        #     # add the df to the dictionary
        #     data_searchlights[center] = searchlight
        #     pbar.update(1)

        # Parallel(n_jobs=-1)(delayed(process_center)(center) for center in vol_searchlight)


        for center in vol_searchlight:
            searchlight_indices = vol_searchlight[center]
            # seachlight should be a 1D array of the voxels (small array of 93 voxels)
            searchlight = {}
            for condition in EVs_data_dict:
                data_condition_searchlight = EVs_data_dict[condition][searchlight_indices]
                
                searchlight[condition] = data_condition_searchlight
            
            # convert the 93 x 20 array to a dataframe
            searchlight = pd.DataFrame(searchlight)
            # add the df to the dictionary
            data_searchlights[center] = searchlight
            pbar.update(1)

    # return the dictionary of the searchlights 
    return data_searchlights

def create_vol_searchlight_dataframe(
        vol_neighbors: list,
        centers: list
) -> pd.DataFrame:
    """
    Create a DataFrame from the volume searchlight  and centers

    The title of each column is the voxel center 
    The rows are the volume neighbours associated with this center

    Parameters:
    - vol_neighbors (list): A list of volume searchlights.
    - centers (list): A list of voxel centers.

    Returns:
    - vol_searchlight (pd.DataFrame): A DataFrame containing the volume searchlights.
    """
    vol_searchlight = {}
    for center, vol_neigh in zip(centers, vol_neighbors):
    # for center, vol_neigh in zip(centers, vol_neighbors):
        vol_searchlight[center] = vol_neigh
    
    return pd.DataFrame(vol_searchlight)    




def get_standard_order():
    """
    Get the standard order of the conditions in the data searchlight.

    Returns:
    - list: A list of the conditions in the standard order.
    """
    conditions = [
        'A1_f', 'A1_b',
        'B1_f', 'B1_b',
        'C1_f', 'C1_b',
        'D1_f', 'D1_b',
        'E1_f', 'E1_b',
        'A2_f', 'A2_b',
        'B2_f', 'B2_b',
        'C2_f', 'C2_b',
        'D2_f', 'D2_b',
        'E2_f', 'E2_b',    
    ]
    return conditions

def sort_data_searchlight(
        data_searchlight: pd.DataFrame,
        conditions_type: str = "two_halves" 
    ) -> pd.DataFrame:
    """
    Sort the data searchlight DataFrame by the conditions.
    
    Parameters:
    - data_searchlight (pd.DataFrame): The DataFrame containing the data searchlight.
    - conditions (list): A list of conditions to sort by.

    Returns:
    - pd.DataFrame: The sorted DataFrame.
    """
    if conditions_type == "two_halves":

        conditions = get_standard_order()

    data_searchlight = data_searchlight[conditions]
    data_searchlight = data_searchlight.reindex(conditions)

    return data_searchlight
def sort_data_searchlight_dict(
        data_searchlight: dict,
        conditions_type: str = "two_halves"
    ) -> dict:
    """
    Sort every DataFrame in the data searchlight dictionary by the conditions.

    Parameters:
    - data_searchlight (dict): A dictionary containing the data searchlight.
    - conditions (list): A list of conditions to sort by.

    Returns:
    - dict: The sorted dictionary.
    """
    for center in data_searchlight:
        data_searchlight[center] = sort_data_searchlight(data_searchlight[center], conditions_type="two_halves")

    return data_searchlight

def get_data_rdms(
    data_searchlight: dict,
    SIZE: str,  
) -> dict:
    """
    Calculate and return the data RDMS (Representational Dissimilarity Matrices) for each center.

    Parameters:
    - data_searchlight (dict): A dictionary containing data for each center.

    Returns:
    - data_rdms_dict (dict): A dictionary containing the data RDMS for each center.
    """


    data_rdms_dict = {}
    with tqdm(total=len(data_searchlight), desc='Creating Data RDMS...') as pbar:
        # def process_center(center):
        #     df = data_searchlight[center]
        #     rdm = get_rdm_from_df(df, SIZE)

        #     # Clear memory after each iteration
        #     del df

        #     return center, rdm

        # num_cores = os.cpu_count()
        # results = Parallel(n_jobs=num_cores)(delayed(process_center)(center) for center in data_searchlight)

        # for center, rdm in results:
        #     data_rdms_dict[center] = rdm
        #     pbar.update(1)


        for center in data_searchlight:
            df = data_searchlight[center]
            rdm = get_rdm_from_df(df, SIZE)

            # Clear memory after each iteration
            del df

            data_rdms_dict[center] = rdm
            pbar.update(1)


    return data_rdms_dict


def dissimilarity_measure(
    v1: np.array, v2: np.array,
    MEASURE: str = "pearson"
)-> float:
    """
    Calculates the similarity measure between two vectors. This is used to make the values in the RDM

    Parameters:
        v1 (np.array): The first vector.
        v2 (np.array): The second vector.
        SIMILARTIY (str, optional): The similarity measure to be used. Defaults to "pearson".

    Returns:
        float: The similarity measure between the two vectors.

    Raises:
        ValueError: If the similarity measure is not "pearson" or "cosine".
    """
    
    if MEASURE == "pearson":
        return 1 - np.corrcoef(v1, v2)[0, 1]
    elif MEASURE == "cosine":
        return - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    else:
        raise ValueError("The similarity measure must be either 'pearson' or 'cosine'.")
    

# def get_rdm_from_df(df: pd.DataFrame,
#                     SIZE: str,
#                     MEASURE: str = "pearson"
#                     ) -> pd.DataFrame:
#     """
#     Calculate the RDM from a DataFrame.

#     Parameters:
#     - df (pd.DataFrame): A DataFrame containing the data for the RDM.

#     Returns:
#     - rdm (pd.DataFrame): A DataFrame containing the RDM.
#     """

#     rdm = pdist(df.T.to_numpy(), metric=dissimilarity_measure)
#     rdm = squareform(rdm)

#     # create the RDM DataFrame
#     rdm = pd.DataFrame(rdm, columns=df.columns, index=df.columns)

#     # sort the dataframe into the two halves order
#     rdm = sort_data_searchlight(rdm, conditions_type="two_halves")

#     if SIZE == "cross_corr":
#         # We are only going to return the comparison between part 1 and part 2 of the task 
#         rdm = rdm.iloc[:10, 10:] + np.transpose(rdm.iloc[10:, :10]) / 2
#         # What is the naming convention for this?

#         return rdm

    
def get_rdm_from_df(df: dict,
                SIZE: str,
                MEASURE: str = "pearson"
                ) -> pd.DataFrame:
    """
    Calculate the RDM from a DataFrame.

    Parameters:
    - df (pd.DataFrame): A DataFrame containing the data for the RDM.

    Returns:
    - rdm (pd.DataFrame): A DataFrame containing the RDM.
    """

    rdm = np.zeros((df.shape[1], df.shape[1]))

    # 1. Sort the order of the df coming in to be the standard order
    # Is this done in the correct order. The main question is whether the RDM EVs are the in the correct order 
    # Are they comparing the correct conditions to each other. If not they it is getting the wrong RDM
    # df = sort_data_searchlight(df, conditions_type="two_halves")
    df = df.loc[:, get_standard_order()]

    ma = df.to_numpy().T
    # # Replace NaN values with 0
    # ma = np.nan_to_num(ma)
    # Normalising the matrix (mean = 0)
    ma = ma - ma.mean(axis=1, keepdims=True)
    # Normalising the matrix (std = 1)
    ma /= np.sqrt(np.einsum('ij,ij->i', ma, ma))[:, None]
    # Computing the dot product of the matrix
    rdm = np.einsum('ik,jk', ma, ma)
    # create the RDM DataFrame
    rdm = pd.DataFrame(rdm, columns=df.columns, index=df.columns)
    # 
    rdm = rdm.iloc[:10, 10:] + np.transpose(rdm.iloc[10:, :10]) / 2

    return rdm


def plot_rdm(
        df: pd.DataFrame,
        idx: int, 
        SIZE: str
    ):
    """
    Plot the RDM (Representational Dissimilarity Matrix) for a given searchlight index.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the RDM data.
    - idx (int): The index of the searchlight.

    Returns:
    - None

    """

    if df[idx].shape[0] != df[idx].shape[1]:
        raise ValueError("The DataFrame must be square to plot the RDM.")
    
    if SIZE == "cross_corr":
        # Plot 10 x 10 matrix
        plt.imshow(df[idx], cmap='plasma')
        plt.xticks(range(10), df[idx].columns, rotation=90)
        plt.yticks(range(10), df[idx].index, rotation=0)
        plt.title(f"RDM for Searchlight {idx}")
        plt.show()

    else:
        plt.imshow(df[idx], cmap='plasma')
        plt.xticks(range(20), df[idx].columns, rotation=90)
        plt.yticks(range(20), df[idx].columns, rotation=0)
        plt.axvline(9.5, color='black')
        plt.axhline(9.5, color='black')
        plt.title(f"RDM for Searchlight {idx}")
        plt.show()


def get_data_rdms_tri(
    data_rdms_dict: dict
) -> pd.DataFrame:
    """
    Convert a dictionary of data RDMS into a DataFrame of lower triangle values.

    Parameters:
        data_rdms_dict (dict): A dictionary containing data RDMS.

    Returns:
        pd.DataFrame: A DataFrame containing the lower triangle values of the data RDMS.
    """

    def get_row_labels(
            column_labels: list,
            index_labels: list
    ): 
        """
        Get the row labels for the data RDMS DataFrame.

        Parameters:
            labels (list): A list of labels.

        Returns:
            list: lower triangle of comparison labels
        """
        label_arr = np.empty((len(column_labels), len(index_labels)), dtype=object)

        for idx1, EV_column in enumerate(column_labels):
            for idx2, EV_index in enumerate(index_labels):
                label_arr[idx1, idx2] = f"{EV_column} vs {EV_index}"

        return label_arr[np.tril_indices(label_arr.shape[0], k=0)]
    

    # NOTE: The labels are the same for all data RDMS so the function should be removed from the for loop
    rdm = data_rdms_dict[list(data_rdms_dict.keys())[0]]
    row_labels = get_row_labels(rdm.columns, rdm.index)

    data_rdms_tri = {}
    # for each RDM 
    for center in data_rdms_dict:
        rdm = data_rdms_dict[center]
        # get the lower triangle of the RDM
        data_rdms_tri[center] = rdm.values[np.tril_indices(rdm.shape[0], k = 0)]
        # create list of correct comparison indicies to label the rows of the data
    # print(row_labels)

    data_rdms_tri = pd.DataFrame(data_rdms_tri)
    data_rdms_tri.index = row_labels
    return data_rdms_tri




def get_EV_dataframe(EVs_path_dict: dict, RDM_VERSION) -> pd.DataFrame:
    """
    Read in dictionary of paths to EVs and load them as numpy arrays into a dictionary
    """
    # create data dictionary

    # create list of tasks in the order they are in the EVs
    EV_tasks = list(EVs_path_dict.keys())

    # remove any tasks from EV_dict_keys that are not conditions to be compared but do have associated images
    for task in EV_tasks:
        if any(substring in task for substring in ['ev_press_EV_EV']):
            EV_tasks.remove(task)

    # Sort the order to be alphabetical
    EV_tasks.sort()

    EVs_data_dict = {}

    # TODO: this EV_tasks must be in the correct order
    # Use the same list to create the model RDm to load in the data dictinoary (to become the data RDM )
    for task_idx, task in enumerate(EV_tasks):
        EVs_data_dict[task] = nib.load(EVs_path_dict[task]).get_fdata()



    return pd.DataFrame(EVs_data_dict)




def create_dummy_searchlight():
    """
    Creates a dummy searchlight DataFrame with random data.

    Returns:
        pd.DataFrame: A DataFrame containing random data for each condition.
    """
    conditions = [
        'A1f', 'A1b',
        'B1f', 'B1b',
        'C1f', 'C1b',
        'D1f', 'D1b',
        'E1f', 'E1b',
        'A2f', 'A2b',
        'B2f', 'B2b',
        'C2f', 'C2b',
        'D2f', 'D2b',
        'E2f', 'E2b',    
    ]
    data_searchlight = {}
    for condition in conditions:
        data_searchlight[condition] = np.random.rand(93)
    return pd.DataFrame(data_searchlight)




def evaluate_model(
        Y,
        X 
        ):
    """
    Evaluates fit of model to data using OLS regression.
    
    Predicting the data from the model
    Y (data) = bX (model) + C

    Parameters
        model: 
        data

    Returns
        tvalues
        betas
        pvalues
    """



    # X = X.reset_index(drop=True)
    X = sm.add_constant(X)
    # print(X.shape, Y.shape)

    # Replace NaN values with 0
    X = np.nan_to_num(X)

    # remove any rows with NaN values 
    nan_filter = np.isnan(X).any(axis=1)

    X = X[~nan_filter]
    Y = Y[~nan_filter]

    est = sm.OLS(Y, X)
    est = est.fit()
    
    # return the tvalues, betas and pvalues
    return est.tvalues[1:], est.params[1:], est.pvalues[1:]


def evaluate_model_parallel(
    data_rdms_tri,
    model_rdms_dict_tri,
    MODEL_TYPE = "replay"
):
    
    
    eval_result = Parallel(n_jobs=-1)(
        delayed(evaluate_model)(
            Y=data_rdms_tri[center],
            X=model_rdms_dict_tri[MODEL_TYPE]
        ) for center in tqdm(data_rdms_tri.columns, desc="Data Searchlights Running")
    )

    return eval_result




def save_RSA_result(
    results_file,
    data_rdms_tri,
    mask,         
    results_directory
    ):
    """
    Saves the results estimates list into the correct maps of the brain

    Parameters
    - results_file: list of tvalues, betas and pvalues as a list of tuples 
    - data_rdms_tri: pd.DataFrame. The data RDMs in upper triangle format. Used to get the centres of the searchlights
    - mask: nilearn.masking.Nifti1Image. The mask of the brain. Used to get the shape of the brain and the affine transform 

    Returns
    - None
    """
    assert len(results_file) == len(data_rdms_tri.columns), "The number of results must be the same as the number of searchlights"



    # get the shape of the brain to be saved into
    x, y, z = mask.shape

    # Create a 1D array of the brain that will store the results
    t_values = np.zeros([x * y * z])
    b_values = np.zeros([x * y * z])
    p_values = np.zeros([x * y * z])
    # for each value in the data_rdms_tri.columns we have the tvalues, betas and pvalues in the eval_result list, inorder
    for idx, centers in enumerate(data_rdms_tri.columns):
        # Unpack each value from the 3 x 1 tuple list and store them in the 1D arrays of the same size as the brain 
        t_values[centers] = results_file[idx][0]
        b_values[centers] = results_file[idx][1]
        p_values[centers] = results_file[idx][2]

    # reshape the 1D arrays to the shape of the brain
    t_values = np.reshape(t_values, mask.shape)
    b_values = np.reshape(b_values, mask.shape)
    p_values = np.reshape(p_values, mask.shape)

    # save the results to a nifti file
    t_values = nib.Nifti1Image(t_values, mask.affine)
    b_values = nib.Nifti1Image(b_values, mask.affine)
    p_values = nib.Nifti1Image(p_values, mask.affine)

    # Create the results directory if it does not exist
    if not os.path.exists(results_directory + '/results'):
        os.makedirs(results_directory + '/results')

    # save results to the correct directory of the brain 
    nib.save(t_values, results_directory + '/results/t_values.nii.gz')  
    nib.save(b_values, results_directory + '/results/b_values.nii.gz')
    nib.save(p_values, results_directory + '/results/p_values.nii.gz')