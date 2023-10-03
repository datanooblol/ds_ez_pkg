import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import math

def check_relationship(data:pd.DataFrame, X:list, y:str, alpha:float, dtype)->dict:
    
    result_dict = defaultdict(list)
    dependent_list = []
    independent_list = []
    for x in X:
        
        if dtype=='categorical':
            table = pd.crosstab(data[y], data[x])
            stat, p, dof, expected = chi2_contingency(table)
        elif dtype=='numerical':
            table = data.groupby(y)[x].apply(list)
            stat, p = f_oneway(*table)
        else:
            pass
        
        if p > alpha:
            independent_list.append(x)
        else:
            dependent_list.append(x)
            
    return {
        'dependent': dependent_list,
        'independent': independent_list,
    }

def check_cat_vs_cat(df:pd.DataFrame, X:list, y:str, alpha:float=0.05)->dict:
    """ Check the association between two categorical features
    Args:
        df (pd.DataFrame) : dataset
        X (list) : the features of interest
        y (str) : the label
        alpha (float) : the cutoff ratio for p-value validation
    Return:
        dict : a dictionary containing dependent and independent list of features
    """
    proxy = df.copy()
    proxy.loc[:, X] = proxy.loc[:, X].fillna('missing')
    return check_relationship(data=proxy, X=X, y=y, alpha=alpha, dtype='categorical')
    
def check_cat_vs_num(df:pd.DataFrame, X:list, y:str, alpha:float=0.05)->dict:
    """ Check whethere there's a similarity between two data distrubuions
    Args:
        df (pd.DataFrame) : dataset
        X (list) : the features of interest
        y (str) : the label
        alpha (float) : the cutoff ratio for p-value validation
    Return:
        dict : a dictionary containing dependent and independent list of features
    """
    proxy = df.copy()
    proxy.loc[:, X] = proxy.loc[:, X].fillna(-1)
    return check_relationship(data=proxy, X=X, y=y, alpha=alpha, dtype='numerical')

def plot_cat_vs_cat(data:pd.DataFrame, X:list, y:str, title:str, ncols:int=2, width:int=16, height:int=3)->None:
    """ Visualize the proportion between categorical and categorical features. On the left is actual values whereas on the right is normalized values.
    Args:
        data (pd.DataFrame) : a lookup from function: check_prevalence
        X (list) : the features of interest
        y (str) : the label
        title (str) : title
        ncols (int) : number of columns represents in the graph
        width (int) : a graph width
        height (int) : a graph height    
    """
    nrows = len(X)
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(width, height*nrows))
    axes = axes.ravel()
    
    fig.suptitle(title, fontsize=35)
    
    for enum, x in enumerate(X):
        res = pd.crosstab(data[x], data[y],).plot(kind='bar', stacked=False, ax=axes[(enum*2)], rot=0)
        res_norm = pd.crosstab(data[x], data[y], normalize='index').plot(kind='bar', stacked=True, ax=axes[(enum*2)+1], rot=0)
        
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])

def get_upper_n_lower(x):
    q1 = x.quantile(.25)
    q3 = x.quantile(.75)
    iqr = q3-q1
    return {
        'upper': q3+(iqr*1.5),
        'lower': q3-(iqr*1.5),
    }

def create_boxplot_data(data:pd.DataFrame, grp_col:str, agg_col:str)->pd.DataFrame:
    res = pd.DataFrame()
    lookup = data.groupby(grp_col).agg(
        upper=pd.NamedAgg(column=agg_col, aggfunc=lambda x: get_upper_n_lower(x)['upper']),
        lower=pd.NamedAgg(column=agg_col, aggfunc=lambda x: get_upper_n_lower(x)['lower']),
    )
    for key in lookup.index:
        mask = True
        mask &= data[grp_col]==key
        mask &= data[agg_col]>=lookup.loc[key, 'lower']
        mask &= data[agg_col]<=lookup.loc[key, 'upper']
        proxy = data.loc[mask,:]
        res = pd.concat([res, proxy])
        
    return res.reset_index(drop=True)
    
def plot_cat_vs_num(data:pd.DataFrame, X:list, y:str, title:str, ncols:int=2, width:int=16, height:int=3)->None:
    """ Visualize the distribution between categorical and numerical features. On the left graph is values with outlier whereas on the right is values without outlier
    Args:
        data (pd.DataFrame) : a lookup from function: check_prevalence
        X (list) : the features of interest
        y (str) : the label
        title (str) : title
        ncols (int) : number of columns represents in the graph
        width (int) : a graph width
        height (int) : a graph height
    """
    nrows = len(X)
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(width, height*nrows))
    axes = axes.ravel()
    
    fig.suptitle(title, fontsize=35)
    
    for enum, x in enumerate(X):
        sns.boxplot(ax=axes[(enum*2)], data=data, x=x, y=y, orient="h")
        cleaned_df = create_boxplot_data(data=data, grp_col=y, agg_col=x).reset_index(drop=True)
        sns.boxplot(ax=axes[(enum*2)+1], data=cleaned_df, x=x, y=y, orient="h")
        
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    
def check_prevalence(data:pd.DataFrame, grp_col:str, agg_col:str, n_round:int=2)->pd.DataFrame:
    """ Create a lookup dataset summarizing the size, occurance and occurance rate or so called prevalance
    Args:
        data (pd.DataFrame) : a dataset
        grp_col (str) : the feature of interest
        agg_col (str) : the occurence of interest
        n_round (int) : round up in n decimal
    Return:
        pd.DataFrame : a lookup of prevalance of grp_col
    """
    return data.groupby(grp_col).agg(
        size=pd.NamedAgg(column=agg_col, aggfunc='count'),
        occurance=pd.NamedAgg(column=agg_col, aggfunc='sum'),
        rate=pd.NamedAgg(column=agg_col, aggfunc=lambda x: (np.sum(x)/len(x))*100),
    ).round(n_round).reset_index()

def plot_prevalence(data:pd.DataFrame, x:str, y1:str, y2:str, title:str, width=10, height=6, alpha=0.3)->None:
    """ Visualize y1=size and y2=occurance rate or so called prevalance in dual axis graph
    Args:
        data (pd.DataFrame) : a lookup from function: check_prevalence
        x (str) : the feature of interest
        y1 (str) : the size of each values
        y2 (str) : the occurence rate of each values
        title (str) : title
        width (int) : a graph width
        height (int) : a graph height
        alpha (float) : a bar opacity
    """
    fig, ax1 = plt.subplots(figsize=(width, height))
    fig.suptitle(title, fontsize=35)
    color1='tab:gray'
    color2='tab:blue'
    ax2 = ax1.twinx()
    ax1.bar(data[x], data[y1], color=color1, alpha=alpha)
    ax1.set_ylabel(y1, color=color1)
    ax2.plot(data[x], data[y2], color=color2, )
    ax2.scatter(data[x], data[y2], color=color2, marker='x')
    ax2.set_ylabel(y2, color=color2)
    for i,j in zip(data[x], data[y2]):
        ax2.annotate(f'{j}%',xy=(i,j))
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])