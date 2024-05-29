"""
Visualisation functions
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style(style='darkgrid')

def plot_RDM_object(RDM_object: np.array, conditions: list = None, title: str = ""):
    """
    Plots the RSA Model
    :param RSA_Model: rsa.RDMs object - This is the matrix of the RSA Model
    :param conditions: list - This is the list of conditions. This is used as the labels
    
    TODO: Write a function that can plot a lot of RDMs stored in one RDM object at once
    """
    # if len(RDM_object.descriptors) != 1:
    #     raise NotImplementedError(f"This function only supports one RDM in the RDMs object. You have got {len(RDM_object.descriptors)} RDMS")
    # else:
    #     pass

    # returns the 3D stack of RDMs. One for each model
    RDM_array = RDM_object.get_matrices()

    # get the number of models
    num_models = RDM_array.shape[0]

    # create a subplot for each model as a square
    # fig, axs = plt.subplots(num_models//2, num_models//2, figsize=(5, 5))

    # for i in range(num_models):
    #     # get the current axis
    #     ax = axs[i//2, i%2]

    #     # plot the RDM
    #     sns.heatmap(RDM_array[i], ax=ax, cmap='plasma')
    #     ax.set_title(RDM_object.descriptors[i])
    #     ax.set_xticks(range(len(conditions)))
    #     ax.set_yticks(range(len(conditions)))
    #     ax.set_xticklabels(conditions, rotation=45, horizontalalignment='right')
    #     ax.set_yticklabels(conditions, horizontalalignment='right')
    #     ax.set_xlabel("Condition 1")
    #     ax.set_ylabel("Condition 2")



    plt.imshow(RDM_array[0], cmap='plasma')
    # add vertical line to plot
    plt.axvline(9.5, color='black', linewidth=2)
    plt.axhline(9.5, color='black', linewidth=2)
    plt.colorbar()
    plt.grid(False)
    plt.title(title)
    # plt.xticks(range(len(conditions)), conditions, rotation=45, horizontalalignment='right')
    # plt.yticks(range(len(conditions)), conditions             , horizontalalignment='right')
    plt.xlabel("Condition 1")
    plt.ylabel("Condition 2")
    plt.show()


if __name__ == "__main__":

    class RDMs:
        def __init__(self, RDMs: np.array, descriptors: list):
            self.RDMs = RDMs
            self.descriptors = descriptors

        def get_matrices(self):
            return self.RDMs
    
    RDMs_object = RDMs(np.random.rand(3, 10, 10), ["Model 1", "Model 2", "Model 3"])

    conditions = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    title = "Random RDMs"
    plot_RDM_object(
        RDM_object=RDMs_object,
        conditions=conditions,
        title=title
    )