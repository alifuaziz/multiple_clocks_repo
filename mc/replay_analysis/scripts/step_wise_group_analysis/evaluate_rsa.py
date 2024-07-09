import pickle
import pandas as pd
from tqdm import tqdm
import os

# import mc.replay_analysis.functions.data_rdms as data_rdms
# import mc.replay_analysis.functions.model_rdms as model_rdms

import data_rdms
import model_rdms

from joblib import Parallel, delayed
from nilearn.image import load_img


def main(    
        **kwargs  
):
    # unpack kwargs

    SUBJECT_DIRECTORY   = kwargs['META_DATA'].get('SUBJECT_DIRECTORY')
    SUB                 = kwargs['META_DATA'].get('SUB')
    RDM_VERSION         = kwargs['META_DATA'].get('RDM_VERSION')
    EVS_TYPE            = kwargs['META_DATA'].get('EVS_TYPE')
    TR                  = kwargs['META_DATA'].get('TR')

    if TR is not None:
        searchlight_data_rdms_file = f"{SUBJECT_DIRECTORY}/analysis/{EVS_TYPE}/TR{TR}/preprocessing/searchlight_data_rdms.pkl"
    else:
        searchlight_data_rdms_file = f"{SUBJECT_DIRECTORY}/analysis/{EVS_TYPE}/preprocessing/searchlight_data_rdms.pkl"

    # load data_searchlight from a pickle file
    with open(searchlight_data_rdms_file, 'rb') as f:
        data_rdms_dict = pickle.load(f)



    if RDM_VERSION == 'replay_nan_off_diag':
        # Set all the off diagonals elements to NaNs
        print("Setting off diagonal elements to NaNs")
        data_rdms_dict = data_rdms.get_data_rdms_nan_off_diag(
            data_rdms_dict = data_rdms_dict
        )

        # convert rdms to vectors for evaluation (that does not need to be modified to be the smae shape as the model   )
        data_rdms_dict_vectors = data_rdms.get_data_rdms_vectors_off_diag(
            data_rdms_dict = data_rdms_dict
        )


        conditions = data_rdms.get_standard_order()

        model_rdms_dict = data_rdms.get_model_rdms(
            conditions = conditions, 
            TYPE = RDM_VERSION, 
        )

        model_rdms_dict = data_rdms.get_data_rdms_vectors_off_diag(
            data_rdms_dict = model_rdms_dict
        )
    else:
        # Keep the RDM the same shape since non will be NaNs
        print("Keeping the RDM the same shape")

        # convert rdms to vectors for evaluation (that does not need to be modified to be the smae shape as the model   )
        # Function for this not written
        data_rdms_dict_vectors = data_rdms.get_data_rdms_vectors_square(
            data_rdms_dict = data_rdms_dict
        )

        # 
        conditions = data_rdms.get_standard_order()

        model_rdms_dict = model_rdms.get_model_rdms(
            conditions = conditions, 
            TYPE = RDM_VERSION, 
        )

        model_rdms_dict = data_rdms.get_data_rdms_vectors_square(
            data_rdms_dict = model_rdms_dict
        )
        pass


    # Load the model rdms
    # load in the model rdm




    eval_result = []
    for searchlight in tqdm(data_rdms_dict_vectors, desc = "Data Searchlights Running"):
        # Evaluate the model
        eval_result.append(data_rdms.evaluate_model(
            Y = model_rdms_dict[RDM_VERSION],                                                      # Model that is being evaluated              
            X = data_rdms_dict_vectors[searchlight]
            )
        )

    # Used to get the metadata from the nii.gz file
    mask = load_img(f"{SUBJECT_DIRECTORY}/anat/{SUB}_T1w_noCSF_brain_mask_bin_func_01.nii.gz")

    if TR is not None:
        results_directory = f"{SUBJECT_DIRECTORY}/analysis/{EVS_TYPE}/TR{TR}/{RDM_VERSION}/results"
    else:
        results_directory = f"{SUBJECT_DIRECTORY}/analysis/{EVS_TYPE}/{RDM_VERSION}/results"

    # Create the results directory if it does not exist
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)


    data_rdms.save_RSA_result(
        results_file = eval_result,
        data_rdms_tri = data_rdms_dict,
        mask = mask,
        results_directory = results_directory,
        RDM_VERSION = RDM_VERSION
    )

    # Save the meta data as a text file to the results directory
    with open(f"{results_directory}/META_DATA.txt", 'w') as f:
        f.write(str(kwargs['META_DATA']))
        # add date and time of analysis completion
        f.write("\nDate of analysis completion:")
        f.write(str(pd.Timestamp.now()))

        