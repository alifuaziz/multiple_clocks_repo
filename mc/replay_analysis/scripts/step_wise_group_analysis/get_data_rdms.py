import pickle
import pandas as pd

from mc.replay_analysis.functions import data_rdms
from mc.replay_analysis.functions import model_rdms



def main(    
        **kwargs  
):
    # unpack kwargs
    SUBJECT_DIRECTORY   = kwargs['META_DATA'].get('SUBJECT_DIRECTORY')
    RDM_VERSION         = kwargs['META_DATA'].get('RDM_VERSION')

    # load data_searchlight from a pickle file
    with open(f"{SUBJECT_DIRECTORY}/analysis/preprocessing/data_searchlight_df.pkl", 'rb') as f:
        data_searchlight = pickle.load(f)

    # main function that will be called

    data_rdms_dict = data_rdms.get_data_rdms(
    data_searchlight = data_searchlight,
    )

    # # convert to triangle vectors for RSA
    # data_rdms_tri = data_rdms.get_data_rdms_tri(
    #     data_rdms_dict = data_rdms_dict
    # )


    # save
    with open(f"{SUBJECT_DIRECTORY}/analysis/{RDM_VERSION}/preprocessing/searchlight_data_rdms.pkl", 'wb') as f:
        pickle.dump(data_rdms_dict, f)

    # with open(f"{SUBJECT_DIRECTORY}/analysis/{RDM_VERSION}/preprocessing/searchlight_data_rdms_tri.pkl", 'wb') as f:
    #     pickle.dump(data_rdms_tri, f)


        