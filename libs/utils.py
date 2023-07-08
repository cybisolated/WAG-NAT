"""Generic helper functions used across codebase."""

import os
import pathlib

import numpy as np
from data_formatters import *

from script_get_data import *


# Generic.
def get_single_col_by_input_type(input_type, column_definition):
    """Returns name of single column.

    Args:
      input_type: Input type of column to extract
      column_definition: Column definition list for experiment
    """

    l = [tup[0] for tup in column_definition if tup[2] == input_type]

    if len(l) != 1:
        raise ValueError('Invalid number of columns for {}'.format(input_type))

    return l[0]


def extract_cols_from_data_type(data_type, column_definition,
                                excluded_input_types):
    """Extracts the names of columns that correspond to a define data_type.

    Args:
      data_type: DataType of columns to extract.
      column_definition: Column definition to use.
      excluded_input_types: Set of input types to exclude

    Returns:
      List of names for columns with data type specified.
    """
    return [
        tup[0]
        for tup in column_definition
        if tup[1] == data_type and tup[2] not in excluded_input_types
    ]


def extract_cols_from_input_type(input_type, column_definition,
                                 excluded_input_types):
    return [
        tup[0]
        for tup in column_definition
        if tup[2] == input_type and tup[2] not in excluded_input_types
    ]


# OS related functions.
def create_folder_if_not_exist(directory):
    """Creates folder if it doesn't exist.

    Argsc
      directory: Folder path to create.
    """
    # Also creates directories recursively
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)


def get_embedding_size(n: int, max_size: int = 100) -> int:
    """
    Determine empirically good embedding sizes (formula taken from fastai).

    Args:
        n (int): number of classes
        max_size (int, optional): maximum embedding size. Defaults to 100.

    Returns:
        int: embedding size
    """
    if n > 2:
        return min(round(1.6 * n**0.56), max_size)
    else:
        return 1


def load_dataset(data_dir, dataset_name):
    if dataset_name == 'ozone':
        # load processed data (pd.DataFrame)
        df_data = load_ozone_data(data_dir)
        # create data formatter w.r.t specific dataset
        formatter = OzoneFormatter()
    elif dataset_name == 'electricity':
        df_data = load_electricity_data(data_dir)
        formatter = ElectricityFormatter()
    elif dataset_name == 'etth1':
        df_data = load_ett_data(data_dir, 'ETTh1.csv')
        formatter = ETTFormatter()
    elif dataset_name == 'etth2':
        df_data = load_ett_data(data_dir, 'ETTh2.csv')
        formatter = ETTFormatter()
    elif dataset_name == 'ettm1':
        df_data = load_ett_data(data_dir, 'ETTm1.csv')
        formatter = ETTFormatter()
    elif dataset_name == 'ettm2':
        df_data = load_ett_data(data_dir, 'ETTm2.csv')
        formatter = ETTFormatter()
    else:
        raise ValueError('not have this dataset')
    return df_data, formatter
