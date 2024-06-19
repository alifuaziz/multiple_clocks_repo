
import pandas as pd
import pickle

from mc.replay_analysis.functions import data_rdms
from mc.analyse.searchlight import get_volume_searchlight
from nilearn.image import load_img

def main(    
        **kwargs  
):
    SUBJECT_DIRECTORY   = kwargs['META_DATA'].get('SUBJECT_DIRECTORY')
    SUB                 = kwargs['META_DATA'].get('SUB')
    EVS_TYPE            = kwargs['META_DATA'].get('EVS_TYPE')
    RDM_VERSION         = kwargs['META_DATA'].get('RDM_VERSION')


    # Load the EVs data



    # Create dictionary to store the data for each EV for both task halves
    # Convert the output of this to be pandas arrays

    EVs_both_halves_dict = {
        '1': None,
        '2': None
    }
    # create new dictionary to store the 2D array of EVs for both task halves
    EVs_both_halves_2d = EVs_both_halves_dict.copy()

        
    EVs_path_dict = data_rdms.get_EV_path_dict(
        subject_directory = SUBJECT_DIRECTORY,
        EVs_type = EVS_TYPE
        )

    # Load in the EVs for the instruction periods from the dictionary of paths
    EVs_data_dict = data_rdms.load_EV_data(
        EVs_path_dict = EVs_path_dict,
    )

    # Unravel the values of each EV
    EVs_data_dict = data_rdms.unravel_EV_data(EVs_data_dict)
    EVs_data_dict = pd.DataFrame(EVs_data_dict)


    # get column names
    column_names = EVs_data_dict.columns.tolist()
    for idx, name in enumerate(column_names):
        column_names[idx] = name[3:7]
        

    # column_names
    EVs_data_dict.rename(columns=dict(zip(EVs_data_dict.columns.tolist(), column_names)), inplace=True)

    conditions = data_rdms.get_standard_order()
    # reorder the columns 
    EVs_data_dict.loc[:, conditions]


    # %%


    mask = load_img(f"{SUBJECT_DIRECTORY}/anat/{SUB}_T1w_noCSF_brain_mask_bin_func_01.nii.gz")
    # print(mask.shape)
    mask_array = mask.get_fdata()
    # print(mask.shape)


    # Get list of voxel centers and their volume neighbours
    centers, vol_neighbors = get_volume_searchlight(
        mask = mask_array,
        radius = 3,
        threshold = 0.5)


    # Deals with searchlights that are not the correct size
    vol_neighbors = data_rdms.resize_neighbors(
        vol_neighbors = vol_neighbors,
        size = 93
    )

    vol_searchlight = data_rdms.create_vol_searchlight_dataframe(
    vol_neighbors = vol_neighbors,
    centers = centers
    )


    data_searchlight = data_rdms.get_data_searchlight(
        vol_searchlight = vol_searchlight,
        EVs_data_dict = EVs_data_dict,
    )

    # Save the vol_searchlight to a pickle file
    with open(f"{SUBJECT_DIRECTORY}/analysis/{RDM_VERSION}/preprocessing/vol_searchlight_df.pkl", 'wb') as f:
        pickle.dump(vol_searchlight, f)

    # Save the data_searchlight to a pickle file
    with open(f"{SUBJECT_DIRECTORY}/analysis/{RDM_VERSION}/preprocessing/data_searchlight_df.pkl", 'wb') as f:
        pickle.dump(data_searchlight, f)
        