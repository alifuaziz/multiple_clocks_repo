import pickle
import pandas as pd
from tqdm import tqdm
import os

import mc.replay_analysis.functions.data_rdms as data_rdms
import mc.replay_analysis.functions.model_rdms as model_rdms

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
    TR                  = kwargs['META_DATA'].get('TR', None)

    # load stuff in
    if TR is not None:
        with open(f"{SUBJECT_DIRECTORY}/analysis/{EVS_TYPE}/TR{TR}/preprocessing/searchlight_data_rdms.pkl", 'rb') as f:
            data_rdms_dict = pickle.load(f)
    else:
        with open(f"{SUBJECT_DIRECTORY}/analysis/{EVS_TYPE}/preprocessing/searchlight_data_rdms.pkl", 'rb') as f:
            data_rdms_dict = pickle.load(f)        



    # convert to triangle vectors for RSA
    # data_rdms_tri = data_rdms.get_data_rdms_tri(
    #     data_rdms_dict = data_rdms_dict
    # )
    # load in the model rdm
    conditions = data_rdms.get_standard_order()

    model_rdms_dict = model_rdms.get_model_rdms(
    conditions = conditions, 
    TYPE = RDM_VERSION, 
    )

    if RDM_VERSION == 'replay_nan_off_diag':
        # Set all the off diagonals elements to NaNs
        data_rdms_dict = data_rdms.get_data_rdms_nan_off_diag(
            data_rdms_dict = data_rdms_dict
        )
        
        if TR is not None:
            searchlight_data_rdms_file = f"{SUBJECT_DIRECTORY}/analysis/{EVS_TYPE}/TR{TR}/preprocessing/searchlight_data_rdms_nan_off_diag.pkl"
        else:
            searchlight_data_rdms_file = f"{SUBJECT_DIRECTORY}/analysis/{EVS_TYPE}/preprocessing/searchlight_data_rdms_nan_off_diag.pkl"

        # save the data_rdms_dict for future use
        with open(searchlight_data_rdms_file, 'wb') as f:
            pickle.dump(data_rdms_dict, f)

        # Load the data_rdms_dict for future use
        with open(searchlight_data_rdms_file, 'rb') as f:
            data_rdms_dict = pickle.load(f)


        # Convert the data rdm to vectors for evaluation
        data_rdms_dict = data_rdms.get_data_rdms_vectors(
            data_rdms_dict = data_rdms_dict
        )

        model_rdms_dict = data_rdms.get_data_rdms_nan_off_diag(
            data_rdms_dict = model_rdms_dict
        )

        model_rdms_dict = data_rdms.get_data_rdms_vectors(
            data_rdms_dict = model_rdms_dict
        )

    # with open(f"{SUBJECT_DIRECTORY}/searchlight_data_rdms_tri.pkl", 'rb') as f:
    #     data_rdms_tri = pickle.load(f)

    # model_rdms_dict_tri = data_rdms.get_data_rdms_tri(
    #     data_rdms_dict = model_rdms_dict
    #     )


    eval_result = []
    for searchlight in tqdm(data_rdms_dict, desc = "Data Searchlights Running"):
        # Evaluate the model
        eval_result.append(data_rdms.evaluate_model(
            Y = model_rdms_dict[RDM_VERSION],                                                      # Model that is being evaluated              
            X = data_rdms_dict[searchlight]
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
    pass