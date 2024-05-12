"""
Replay Models

Here the matrices that define the different condition for replay (matricies) are created. 


"""
# Importing Libraries
import numpy as np

import mc.replay_analysis.visualise as v


def initilaize_EVs() -> list:
    """
    Initialize the conditions that are used to create the RDMs
    
    :return: list of conditions
    """
    EVs = ['A1_forw', 'A1_backw', \
           'B1_forw', 'B1_backw', \
           'C1_forw', 'C1_backw', \
           'D1_forw', 'D1_backw', \
           'E1_forw', 'E1_backw', \
           'A2_forw', 'A2_backw', \
           'B2_forw', 'B2_backw', \
           'C2_forw', 'C2_backw', \
           'D2_forw', 'D2_backw', \
           'E2_forw', 'E2_backw']
    
    return EVs

def create_execution_RSM(conditions: list) -> np.array:
    """
    Creates an execution RDM model
    
    Confounders that this model controls for:
    - The two task halves
       - Since the reverse execution is in the second task half, the model implicitly controls for the task half
    - Difficulty of doing the forward and backward version of recall.
        i.e. We assume that doing the forward version of recall is easier than doing the backward version of the mirror task, even though the task iself, is the same

    :param conditions: list of different conditions are compared that represent the different conditions given to to participant during the study
    
    :return: RDM_Models matrix (np.ndarray). This is the matrix of similarities between the different conditions
    """
    # initialize RDM matrix
    RSM_Model = np.zeros((len(conditions), len(conditions)))

    # the condition being compared to all other conditions (including itself)
    for idx1, condition1 in enumerate(conditions):

        for idx2, condition2 in enumerate(conditions):
        
            if condition1[0] == condition2[0]: # A, B, C, D, E
                # they are both the same pattern of coins
                if condition1[1] == condition2[1]: # 1, 2
                    # they are the same task half. In different task halves the excuation (A1 is the reverse of the execution A2)
                    if condition1[2] == condition2[2]:
                        # they are the same instructions directions
                        RSM_Model[idx1, idx2] = +1
                    else:
                        # they are different instructions directions
                        RSM_Model[idx1, idx2] = -1
                else:
                    # they are different  task halves, the execution is the reverse of the other task half
                    if condition1[2] == condition2[2]: # f, b
                        # they are the same instructions directions
                        RSM_Model[idx1, idx2] = -1 # i.e. A1f is the reversed executution of A2f
                    else:
                        RSM_Model[idx1, idx2] = +1 # i.e. A1f is the same execution as A2b

            else:
                # they are different patterns of coins
                if condition1[1] == condition2[1]:
                    # they are the same task half
                    if condition1[2] == condition2[2]:
                        # they are the same instructions directions
                        RSM_Model[idx1, idx2] = -1/4
                    else:
                        RSM_Model[idx1, idx2] = +1/4
                    pass
                else:
                    # they are different task halves
                    if condition1[2] == condition2[2]:
                        # they are the same instructions directions
                        RSM_Model[idx1, idx2] = -1/4
                    else:
                        RSM_Model[idx1, idx2] = +1/4    
                pass                    
    
    return RSM_Model

def create_instruction_RSM(conditions: list) -> np.array:
    """
    Creates an instruction RDM model
    
    This compares the similiarity of the period given the to participant for the different conditions
    
    :param conditions: list of different conditions are compared that represent the different conditions given to to participant during the study

    :return: RDM_Models matrix (np.ndarray). This is the matrix of similarities between the different conditions
    """

    RSM_Model = np.zeros((len(conditions), len(conditions)))

    for idx1, condition1 in enumerate(conditions):

        for idx2, condition2 in enumerate(conditions):
        
            if condition1[0] == condition2[0]: # A, B, C, D, E
                # they are both the same pattern of coins
                if condition1[1] == condition2[1]: # part 1 or 2
                    # they are the same task half
                    if condition1[2] == condition2[2]: # f, b
                        pass


if __name__ == "__main__":
    # Define the conditions
    conditions = initilaize_EVs()
    # print(conditions)
    RDM_Model = create_execution_RSM(conditions)
    v.plot_RSA_Model(RDM_Model, conditions, title="Execution RDM")
    pass