import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.inspection import PartialDependenceDisplay, partial_dependence

def plot_contribution(model, 
                      data:pd.DataFrame, features:list, 
                      row:int, col:int, title:str, width:int=4, height:int=3, 
                      categorical_features:list=None, save:str=None)->None:
    """
    Args:
        model () : a sklearn model
        data (pd.DataFrame) : dataset for model prediction
        features (list) : features of analysis
        row (int) : a number of rows represents in the graph
        col (int) : a number of columns represents in the graph
        title (str) : a graph name
        width (int) : a graph width
        height (int) : a graph height
        categorical_features (list) : list of boolean matching with features
        save (str) : a namefile that will be saved ending with *.png
    """
    fig, axes = plt.subplots(row, col, figsize=(width*col,height*row), sharey=True)
    axes = axes.ravel()

    fig.suptitle(title, fontsize=35)    
    
    for enum, feature in enumerate(features):
        PartialDependenceDisplay.from_estimator(model, data, features=[feature], ax=axes[enum], categorical_features=None)

    fig.tight_layout(rect=[0, 0.03, 1, 0.98])

    if save!=None:
        fig.savefig(save)