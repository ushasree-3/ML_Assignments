"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np


def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    # print(y)
    if y.dtypes.name!='category':
        # print(True)
        return True
    else:
        # print(False)
        return False
    

def MSE(series: pd.Series):
    series_MSE = 0
    mean = np.mean(series)
    for i in series:
        series_MSE += (mean - i)**2
    return series_MSE

def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    entropy = 0
    labels = np.unique(Y)
    for label in labels:
        p = len(Y[Y==label])/len(Y)
        entropy += -p*(np.log2(p))
    
    return entropy

def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    gini_index = 1
    labels = np.unique(Y)
    for label in labels:
        p = len(Y[Y==label])/len(Y)
        gini_index -= p**(2)

    return gini_index


def information_gain(Y: pd.Series, attr: pd.Series, mode="Entropy") -> tuple:
    """
    Function to calculate the information gain
    """
    dict1 = {'attr': attr, 'Y': Y}
    df = pd.concat(dict1, axis=1)     # concatinating attributes and Y in a dataframe

    # print('here')
    dict1 = {1: attr, 2: Y}
    df = pd.concat(dict1, axis=1)     # concatinating attributes and Y in a dataframe
    sort_df = df.sort_values(1).reset_index(drop=True)

    if not check_ifreal(Y) and check_ifreal(attr) and mode=="information_gain": # real input, discrete output, entropy  
        # print('here')
        # print(df)
        
        max_gain = -float('inf')
        best_split = 0

        for ind in sort_df.index:
            if ind==0:
                continue
            split = (sort_df[1][ind] + sort_df[1][ind-1])/2
            df1 = sort_df[sort_df[1]<split].reset_index()
            df2 = sort_df[sort_df[1]>=split].reset_index()
            y1 = df1.iloc[:, 2]
            y2 = df2.iloc[:, 2]
            gain = entropy(Y) - ((entropy(y1)*len(y1) + entropy(y2)*len(y2))/len(Y))

            if gain>=max_gain:
                best_split = split
                max_gain = gain

        
        return best_split, max_gain
    
    elif not check_ifreal(Y) and check_ifreal(attr) and mode=="gini_index": # real input, discrete output, gini_index  
        # print('here')
        # print(df)
        
        max_gini = -float('inf')
        best_split = 0

        for ind in sort_df.index:
            if ind==0:
                continue
            split = (sort_df[1][ind] + sort_df[1][ind-1])/2
            df1 = sort_df[sort_df[1]<split].reset_index()
            df2 = sort_df[sort_df[1]>=split].reset_index()
            y1 = df1.iloc[:, 2]
            y2 = df2.iloc[:, 2]
            gini = (gini_index(y1)*(len(y1)) + gini_index(y2)*(len(y2)))/len(Y)
            if gini>=max_gini:
                best_split = split
                max_gini = gini

        
        return best_split, max_gini
    
    elif check_ifreal(Y) and check_ifreal(attr):
        max_var = -float('inf')
        best_split = 0

        for ind in sort_df.index:
            if ind==0:
                continue
            split = (sort_df[1][ind] + sort_df[1][ind-1])/2
            df1 = sort_df[sort_df[1]<split].reset_index()
            df2 = sort_df[sort_df[1]>=split].reset_index()
            y1 = df1.iloc[:, 2]
            y2 = df2.iloc[:, 2]
            var = (np.var(y1)*(len(y1) + np.var(y2)*len(y2))/len(Y))
            if var>=max_var:
                best_split = split
                max_var = var

        
        return best_split, max_var
    
    elif check_ifreal(Y) and not check_ifreal(attr):
        dist_target = attr.value_counts(normalize=True)
        normalized_target = dist_target/dist_target.sum()
        IG = MSE(Y)
        for value, p in normalized_target.items():
            series = Y[attr == value]
            IG = IG - p * MSE(series)
        return (IG,None)
    elif not check_ifreal(Y) and not check_ifreal(attr) and mode=="Entropy":
        dist_target = attr.value_counts(normalize=True)
        normalized_target = dist_target/dist_target.sum()
        IG = entropy(Y)
        for value, p in normalized_target.items():
            series = Y[attr == value]
            IG = IG - p * entropy(series)
        return (IG,None)
    elif not check_ifreal(Y) and not check_ifreal(attr) and mode=="gini_index":
        dist_target = attr.value_counts(normalize=True)
        normalized_target = dist_target/dist_target.sum()
        IG = gini_index(Y)
        for value, p in normalized_target.items():
            series = Y[attr == value]
            IG = IG - p * gini_index(series)
        return (IG,None)


    pass

def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion="information_gain"):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """
    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).
    if (check_ifreal(y) and check_ifreal(X.iloc[:,0])) or (not check_ifreal(y) and check_ifreal(X.iloc[:,0])):
        # print(check_ifreal(X.iloc[:0]), print(X[:,0]))
        max_info_gain = -float("inf")
        best_split = False
        column_req = None
        for column in X:
            attr = pd.Series(X[column])
            split, info_gain = information_gain(y, attr, criterion)
            if(info_gain > max_info_gain):
                max_info_gain = info_gain
                best_split = split
                column_req = column
        return column_req, best_split
    
    elif check_ifreal(y) and not any(check_ifreal(X[j]) for j in X) and criterion == "information_gain":
        IG_list = {}
        for i in X:
            IG_list[i] = information_gain(y, X[i])[0]
        IGvalues = list(IG_list.values())
        IGkeys = list(IG_list.keys())
        return IGkeys[IGvalues.index(max(IGvalues))]
    
    elif not check_ifreal(y) and not any(check_ifreal(X[j]) for j in X): #dicrete input and output
        C = "Entropy"
        if criterion == "information_gain":
            pass
        else:
            C = "gini_index"
        IG_list = {}
        for i in X:
            IG_list[i] = information_gain(y, X[i], mode=C)[0]
        IGvalues = list(IG_list.values())
        IGkeys = list(IG_list.keys())
        return IGkeys[IGvalues.index(max(IGvalues))]        
    pass