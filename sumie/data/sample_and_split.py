import pandas as pd
import numpy as np
from typing import List, Tuple
from itertools import accumulate
from sklearn.model_selection import train_test_split
import glob


def default_sample(df : pd.DataFrame, num_sample : int=None, ratio_sample: float=1.0, random_seed=343) -> pd.DataFrame:
    '''
    Randomly sample rows from input data frame.
    '''
    if(num_sample is not None): 
        ratio_sample = None

    return df.sample(n=num_sample, frac=ratio_sample, random_state=random_seed)

def shuffle(df : pd.DataFrame, random_seed=343) -> pd.DataFrame:
    '''
    Shuffle rows of dataframe.
    '''
    return default_sample(df, ratio_sample=1.0, random_seed=random_seed)

def default_split(df : pd.DataFrame, ratios: List[float]=[1.0], random_seed=343) -> Tuple[pd.DataFrame]:
    '''
    Split dataframe into subsets based on the list of ratios provided. 
    '''
    num_splits = len(ratios)
    ratio_sum = sum(ratios)
    
    if ratio_sum > 1.0: 
        ratios = [r/ratio_sum for r in ratios]
    elif ratio_sum < 1.0: 
        remainder = 1 - ratio_sum
        ratios.append(remainder)

    split_indices = [(frac*len(df)).astype(int) for frac in accumulate(ratios[:-1])]

    return np.split(df, split_indices)[:num_splits]


def merge(dfs : List[pd.DataFrame]) -> pd.DataFrame:
    '''
    Merge list of dataframes into a single dataframe given that they share the same schema.
    ''' 
    return pd.concat(dfs)

def sample_to_smallest_class(df: pd.DataFrame, class_col: str) -> pd.DataFrame:
    ''' 
    Outputs a dataframe where each class in column has n datapoints where n is the number of datapoints
    of the least represented class in the input df
    ''' 
    class_counts = df[class_col].value_counts(ascending=True)
    unique_classes = df[class_col].unique()
    num_data_smallest_class = class_counts.iloc[0]
    return merge([default_sample(df[df[class_col] == c], num_sample=num_data_smallest_class) for c in unique_classes])

def stratified_split(df : pd.DataFrame, ratios: List[float], stratify_col : str, random_seed=343) -> Tuple[pd.DataFrame]:
    '''
    Split dataframe into subsets based on the list of ratios provided, 
    stratified according to the stratify column provided. 
    '''
    num_splits = len(ratios)
    ratio_sum = sum(ratios)
    
    if ratio_sum > 1.0: 
        ratios = [r/ratio_sum for r in ratios]
    elif ratio_sum < 1.0: 
        remainder = 1 - ratio_sum
        ratios.append(remainder)
    
    splits = list()

    remaining_strat = df[stratify_col]
    remaining_data = df
    remaining_data_ratio = 1.0

    for i in range(len(ratios[:-1])):
        current_split_ratio = ratios[i]/remaining_data_ratio 
        current_split, remaining_data, current_strat, remaining_strat = train_test_split(remaining_data, remaining_strat, 
            train_size=current_split_ratio, random_state=random_seed, stratify=remaining_strat)
        remaining_data_ratio = remaining_data_ratio - ratios[i]
        splits.append(current_split)
    splits.append(remaining_data)

    return tuple(splits[:num_splits])


def load(file_path : str) -> pd.DataFrame:
    '''
    Read in csv as dataframe
    ''' 
    return pd.read_csv(file_path)

def save(df : pd.DataFrame, file_path: str):
    '''
    Save dataframe as csv to specified file_path. 
    '''
    df.to_csv(file_path, index=False, encoding='utf-8-sig')