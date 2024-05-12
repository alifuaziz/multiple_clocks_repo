"""
Visualisation functions
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style(style='darkgrid')

def plot_RSA_Model(RSA_Model: np.array, conditions: list, title: str = ""):
    """
    Plots the RSA Model
    :param RSA_Model: np.array - This is the matrix of the RSA Model
    :param conditions: list - This is the list of conditions. This is used as the labels
    """
    plt.imshow(RSA_Model, cmap='plasma')
    # add vertical line to plot
    plt.axvline(9.5, color='black', linewidth=2)
    plt.axhline(9.5, color='black', linewidth=2)
    plt.colorbar()
    plt.grid(False)
    plt.title(title)
    plt.xticks(range(len(conditions)), conditions, rotation=45, horizontalalignment='right')
    plt.yticks(range(len(conditions)), conditions             , horizontalalignment='right')
    plt.xlabel("Condition 1")
    plt.ylabel("Condition 2")
    plt.show()


