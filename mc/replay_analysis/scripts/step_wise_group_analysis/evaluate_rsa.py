import pickle
import pandas as pd
from tqdm import tqdm

from mc.replay_analysis.functions import data_rdms
from mc.replay_analysis.functions import model_rdms

from joblib import Parallel, delayed


def main(    
        **kwargs  
):
    # unpack kwargs

    SUBJECT_DIRECTORY   = kwargs['META_DATA'].get('SUBJECT_DIRECTORY')
    SUB                 = kwargs['META_DATA'].get('SUB')
    # load stuff in

    with open(f"{SUBJECT_DIRECTORY}/searchlight_data_rdms.pkl", 'rb') as f:
        data_rdms_dict = pickle.load(f)


    # convert to triangle vectors for RSA
    data_rdms_tri = data_rdms.get_data_rdms_tri(
        data_rdms_dict = data_rdms_dict
    )



    # with open(f"{SUBJECT_DIRECTORY}/searchlight_data_rdms_tri.pkl", 'rb') as f:
    #     data_rdms_tri = pickle.load(f)



    # load in the model rdm
    conditions = data_rdms.get_standard_order()

    model_rdms_dict = model_rdms.get_model_rdms(
    conditions = conditions, 
    TYPE = 'replay', 
    SIZE = 'cross_corr')


    model_rdms_dict_tri = data_rdms.get_data_rdms_tri(
        data_rdms_dict = model_rdms_dict
        )


    eval_result = []
    for searchlight in tqdm(data_rdms_tri.columns, desc = "Data Searchlights Running"):
        # Evaluate the model
        eval_result.append(data_rdms.evaluate_model(
            Y = model_rdms_dict_tri['replay'],                                                      # Model that is being evaluated              
            X = data_rdms_tri[searchlight]
            )
        )
        # Save to the nii.gz files
        data_rdms.save_RSA_result(
        eval_result = eval_result,
        searchlight = searchlight,
        subject_directory = SUBJECT_DIRECTORY
        )

    # # Evaluate the model
    # eval_result = Parallel(n_jobs=1)(delayed(data_rdms.evaluate_model)(
    #     Y = model_rdms_dict_tri['replay'],                                                      # Model that is being evaluated              
    #     X = data_rdms_tri[searchlight]
    #     ) for searchlight in tqdm(data_rdms_tri.columns, desc = "Data Searchlights Running"))   # (from the list of data RDMs)



    # Save to the nii.gz files
    data_rdms.save_RSA_result(
    eval_result = eval_result,
    searchlight = data_rdms_tri.columns,
    subject_directory = SUBJECT_DIRECTORY
    )


    pass