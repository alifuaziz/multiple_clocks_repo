"""
Utility functions for replay analysis
"""
import numpy as np
from collections import defaultdict

def similarity_measure(vector1: np.array, vector2: np.array, TYPE) -> float:
    """
    Calculate the similiarty between two vectors
    """
    if TYPE == 'cosine':
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    elif TYPE == 'euclidean':
        return np.linalg.norm(vector1 - vector2)
    else:
        raise ValueError(f'TYPE {TYPE} not recognized')
    

def RSM_to_RDM(RSM: np.array) -> np.array:
    """
    Converts a Representational Similarity Matrix (RSM) to a Representational Dissimilarity Matrix (RDM)
    """
    return 1 - RSM
    

def reverse_nested_dict(dict_to_reverse):
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

   
def flatten_nested_dict(nested_dict):
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
 
def print_stuff(string_input):
    """
    This function prints the input string
    """
    print(string_input)
   

if __name__ == "__main__":
    print("Running utils.py as main file")
    pass