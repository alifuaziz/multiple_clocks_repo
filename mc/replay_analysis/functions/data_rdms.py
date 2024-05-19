"""
Alif's Functions for RSA
"""
# Standard Libraries
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
from pathlib import Path



# 
import nibabel as nib 

# RSA specific libraries
from rsatoolbox.rdm import RDMs as RDMs_object

# Multiple Clocks Repositiory
import mc
import mc.analyse.analyse_MRI_behav     as analyse_MRI_behav
import mc.analyse.extract_and_clean     as extract_and_clean
import mc.simulation.predictions        as predictions
import mc.simulation.RDMs               as RDMs
import mc.replay_analysis.functions.utils      as utils
import mc.replay_analysis.functions.visualise  as v
import mc.replay_analysis.functions.model_rdms as functions_model_rdms



"""
Model RSA functions
"""

"""
Different functions for different models
"""


def create_matrix_replay(EVs):
    """
    Returns the RDM for the replay model
    """
    # create the empty RDM array
    rdm_array = np.zeros((len(EVs), len(EVs)))

    # For first condition
    for EV1_idx, EV1 in enumerate(EVs):
        # for second condition
        for EV2_idx, EV2 in enumerate(EVs):
            # compare the conditions
            
            if EV1[0] == EV2[0]: #(A, B, C, D, E) (A == A)
                # the set of rewards are the same

                if EV1[1] == EV2[1]: # Task half (1 or 2) (A1 == A1)
                    # Forwards v.s. backwards

                    if EV1[3] == EV2[3]: # Forwards v.s. backwards 
                        # 
                        pass

                    else:
                        pass
            
                else :# (A1, A2)
                    if EV1[3] == EV2[3]: #(A1F, A2F)
                        rdm_array[EV1_idx, EV2_idx] = -1
                    else: # (A1F, A2B)
                        rdm_array[EV1_idx, EV2_idx] = +1
            else: #(A, B)

                if EV1[1] == EV2[1]: # A

                    if EV1[3] == EV2[3]: # (A1F, B1F)
                        rdm_array[EV1_idx, EV2_idx] = +1/4
                    
                    else: # (A1F, B1B)
                        rdm_array[EV1_idx, EV2_idx] = -1/4

                else:
                    pass


    return rdm_array


"""
Function for making the model RDM from the functions above in the correct format

"""

def task_similarity_matrix(
        configs_dict: dict,
        model: list = "replay",
        RDM_dir: str = None,
        VISUALISE: bool = False
):
    """
    Param
        configs_dict: dict

    Returns
        model_RDM_dict: rsa.RDMs object for one model

    """
    
    # Get the order of the tasks from the configs dictionary
    sorted_keys_dict = extract_and_clean.order_task_according_to_rewards(configs_dict)

    # list of conditions
    EVs = list(sorted_keys_dict['1']) + list(sorted_keys_dict['2'])
    
    if model == "replay":
        # Create the RDMs for the replay analysis
        replay_RDM = create_matrix_replay(EVs)
    else: 
        # There is space here to add more models with new functions
        raise ValueError("Model not found.")

    # stack of RDMs for each condition. in this case, only one RDM. the "replay" one
    rdm_stack = np.stack([replay_RDM])

    # a dictonary containing all the model desciptors
    model_RDM_descripter = {} 
    model_RDM_descripter['replay'] = "Replay Model"

    # Create the RDM object
    replay_RDM_object = RDMs_object(
        dissimilarities = rdm_stack,
        dissimilarity_measure = 'Arbitrary',
        descriptors = model_RDM_descripter,
        # pattern_descriptors =
    )

    if RDM_dir is not None:
        # Save the RDMs object to a pickle file
        with open(f"{RDM_dir}/replay_RDM_object.pkl", 'wb') as file:
            pickle.dump(replay_RDM_object, file)

    if VISUALISE == True:

        # Visualise the RDMs
        v.plot_RDM_object(replay_RDM_object, 
                          title = list(model_RDM_descripter.values())[0],
                          conditions = EVs,)

    return replay_RDM_object



def load_model_RDMs(
        RDM_dir: str,
    ):
    """
    Load in the model_RDMs object from a pickle file
    """
    with open(f"{RDM_dir}/replay_RDM_object.pkl", 'rb') as file:
        replay_RDM_object = pickle.load(file)

    return replay_RDM_object




def get_model_RDM(
        configs_dict: dict,
        USE_NEURON_MODEL = False, 
    ) -> dict: 
    """
    Param
        USE_NEURON_MODEL: bool. Decides if the neuron (vector) model is used or if the RDM is created bespoke

    Returns
        rdm: rsa.rdm.rdms.RDMs Class with the correct RDM within in
    """

    configs_order = {
        "1": configs_dict["1"].keys(),
        "2": configs_dict["2"].keys()
    }

    
    
    # rdm = RDMs(
    #     dissimilarities = RDM
    # )

    # return rdm

if __name__ == '__main__':
    pass