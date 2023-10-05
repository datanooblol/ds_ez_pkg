import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from matplotlib.collections import EventCollection

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

def manual_pdp(model, X:pd.DataFrame, features:list, target_feature:str)->None:
    val_list = X.loc[:, target_feature].sort_values().unique()
    proxy = X.loc[:, list(set(features) - set([target_feature]))].copy()
    proxy = pd.concat([proxy.assign(**{target_feature:i}) for i in val_list], axis=0, ignore_index=True).loc[:, features]
    res = pd.DataFrame({
        f'{target_feature}':proxy.loc[:,f'{target_feature}'], 
        'Partial dependence':model.predict_proba(proxy)[:,1]
    }).groupby(f'{target_feature}')[['Partial dependence']].mean().reset_index().plot.line(x=target_feature, y='Partial dependence')


#-----[the above codes are now deprecated]-----#


#-----[Use this code below]-----#    
    
def get_pdp(model, X:pd.DataFrame, features:list, target_feature:str)->dict:
    """This function will calculate the partial dependence value of the target feature
    Args:
        model () : a sklearn model
        X (pd.DataFrame) : a transformed dataset
        features (list) : input features
        target_feature : the feature of interest
    Return:
        dict : the dictionary contains: df=result, collection=decile of the target_feature, min_pdp=the minimun partial dependence value
    """
    val_list = X.loc[:, target_feature].sort_values().unique()
    proxy = X.loc[:, list(set(features) - set([target_feature]))].copy()
    proxy = pd.concat([proxy.assign(**{target_feature:i}) for i in val_list], axis=0, ignore_index=True).loc[:, features]
    res = pd.DataFrame({
        f'{target_feature}':proxy.loc[:,f'{target_feature}'], 
        'Partial dependence':model.predict_proba(proxy)[:,1]
    }).groupby(f'{target_feature}')[['Partial dependence']].mean().reset_index()
    return {
        'df':res,
        'collection':X[target_feature].quantile(np.arange(.1,.91,0.1)).values,
        'min_pdp':res['Partial dependence'].min()
    }

def plot_pdp(model, X:pd.DataFrame, features:list, target_features:list, 
             title:str, 
             nrows:int, ncols:int, width:int=3, height:int=2, save:str=None):
    """This function will plot the partial dependency of target_features
    Args:
        model () : a sklearn model
        X (pd.DataFrame) : a transformed dataset
        features (list) : input features
        target_features (list) : the features of interest
        title (str) : a graph title
        nrows (int) : number of rows in subplot
        ncols (int) : number of columns in subplot
        width (int) : the width of each graph
        height (int) : the height of each graph
        save (str) : a namefile that will be saved ending with *.png
    """
    n_features = len(target_features)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(width*ncols, height*nrows))
    fig.suptitle(title, fontsize=35)    
    axes = axes.ravel()    
    for enum, feat in enumerate(target_features):

        pdp_dict = get_pdp(model=model, X=X, features=features, target_feature=feat)
        plot_df = pdp_dict['df']
        x_collection = pdp_dict['collection']
        min_pdp = pdp_dict['min_pdp']
        x_collection = EventCollection(x_collection, color='black', linelength=min_pdp*2, linewidth=1, lineoffset=0)
        axes[enum].plot(*[plot_df[col] for col in plot_df.columns])
        axes[enum].add_collection(x_collection)
        axes[enum].set_xlabel(f'{feat}')
        axes[enum].set_ylabel(f'Partial dependence')
        
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    
    if save!=None:
        fig.savefig(save)
