"""
Utility functions for replay analysis

@Author: Alif
"""
import numpy as np
from collections import defaultdict

def similarity_measure(
        condiiton_1: np.array,
        condition_2: np.array, 
        TYPE = "cosine"
        ) -> int:
    """
    Compute the similarity measure for two different conditions. This will eventually fill a similarity matrix
    """

    if TYPE == "cosine":
        similarity = np.dot(condiiton_1, condition_2) / (np.linalg.norm(condiiton_1) * np.linalg.norm(condition_2))
    elif TYPE == "euclidean":
        similarity = np.linalg.norm(condiiton_1 - condition_2)
    elif TYPE == "pearson":
        similarity = np.corrcoef(condiiton_1, condition_2)[0, 1]

    else: 
        raise NotImplementedError(f"Similarity measure {TYPE} not implemented")    

    return similarity
    

def create_similarity_matrix(
        neuron_matrix: np.array, 
        TYPE: str
    ) -> np.array:
    """
    Create a similarity matrix from the data
    """
    no_conditions = neuron_matrix.shape[0]
    sim_matrix = np.zeros((no_conditions, no_conditions))
    for condition1 in range(no_conditions):
        for condition2 in range(no_conditions):
            sim_matrix[condition1, condition2] = similarity_measure(
                neuron_matrix[condition1],
                neuron_matrix[condition2], 
                TYPE)
            
    return sim_matrix

def RSM_to_RDM(RSM: np.array) -> np.array:
    """
    Converts a Representational Similarity Matrix (RSM) to a Representational Dissimilarity Matrix (RDM)
    """
    return 1 - RSM
    

def reverse_nested_dict(dict_to_reverse: dict) -> dict:
    """
    Reverses the nesting of the dictionary by swapping the keys of the inner and outer dictionaries
    e.g.
    {
        'key1': {'key2': 'value1'},
        'key3': {'key4': 'value2'}
    } 
    becomes
    {
        'key2': {'key1': 'value1'},
        'key4': {'key3': 'value2'}
    }

    """
    flipped = defaultdict(dict)
    for key, value in dict_to_reverse.items():
        for key2, value2 in value.items():
            flipped[key2][key] = value2

    # convert the defaultdict to a regular dictionary
    return dict(flipped)

   
def flatten_nested_dict(
        nested_dict: dict
        ) -> dict:
    """
    :param nested_dict: A dictionary with nested dictionaries

    :return: A dictionary with the nested dictionaries flattened
    """

    flattened_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, dict):  # If the value is a dictionary, extend the flat dictionary with its items
            flattened_dict.update(value)
        else:
            flattened_dict[key] = value
    return flattened_dict
 
def print_stuff(string_input: str):
    """
    This function prints the input string
    """
    print(string_input)
   

def get_lower_triangle(matrix: np.array, k: int = 0) -> np.array:
    """
    Get the lower triangle of a matrix. Return it in vector form.

    By default, the diagonal is included. To exclude the diagonal, set k=-1.

    Example:
    matrix = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    get_lower_triangle(matrix) -> [1, 4, 5, 7, 8, 9]

    """
    return np.atleast_2d(matrix[np.tril_indices(matrix.shape[0])])


if __name__ == "__main__":
    print("Running utils.py as main file")
    
    matrix = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    print(get_lower_triangle(matrix))

    pass