# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom formatting functions for Electricity dataset.

Defines dataset specific column definitions and data transformations. Uses
entity specific z-score normalization.
"""

import data_formatters.base
import libs.utils as utils
import pandas as pd
import sklearn.preprocessing

GenericDataFormatter = data_formatters.base.GenericDataFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes


class ETTHFormatter(GenericDataFormatter):
    """Defines and formats data for the electricity dataset.

    Note that per-entity z-score normalization is used here, and is implemented
    across functions.

    Attributes:
      column_definition: Defines input and data type of column used in the
        experiment.
      identifiers: Entity identifiers used in experiments.
    """

    _column_definition = [
        ('id', DataTypes.REAL_VALUED, InputTypes.ID),
        ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.TIME),

        ('HUFL', DataTypes.REAL_VALUED, InputTypes.TARGET),
        ('HULL', DataTypes.REAL_VALUED, InputTypes.TARGET),
        ('MUFL', DataTypes.REAL_VALUED, InputTypes.TARGET),
        ('MULL', DataTypes.REAL_VALUED, InputTypes.TARGET),
        ('LUFL', DataTypes.REAL_VALUED, InputTypes.TARGET),
        ('LULL', DataTypes.REAL_VALUED, InputTypes.TARGET),
        ('OT', DataTypes.REAL_VALUED, InputTypes.TARGET),

        ('hour', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        # ('day', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        # ('day_of_week', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        # ('month', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        # ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT)
    ]

    def __init__(self):
        """Initialises formatter."""

        self.identifiers = None
        self._real_scalers = None
        self._cat_scalers = None
        self._target_scaler = None
        self._num_classes_per_cat_input = None

    def split_data(self, df, L_in, valid_start=360, test_start=480):
        """Splits data frame into training-validation-test data frames.

        This also calibrates scaling object, and transforms data for each split.

        Args:
          df: Source data frame to split.
          valid_boundary: Starting year for validation data
          test_boundary: Starting year for test data

        Returns:
          Tuple of transformed (train, valid, test) data.
        """

        index = df['days_from_start']
        train = df.loc[index < valid_start]
        valid = df.loc[(index >= valid_start - (L_in // 24)) & (index < test_start)]
        test = df.loc[index >= test_start - (L_in // 24)]

        self.set_scalers(train)

        return (self.transform_inputs(data) for data in [train, valid, test])

    def set_scalers(self, df):
        """Calibrates scalers using the data supplied.

        Args:
          df: Data to use to calibrate scalers.
        """

        column_definitions = self.get_column_definition()
        target_column = utils.extract_cols_from_input_type(InputTypes.TARGET, column_definitions, {})

        # Format real scalers
        real_inputs = utils.extract_cols_from_data_type(DataTypes.REAL_VALUED,
                                                        column_definitions,
                                                        {InputTypes.ID, InputTypes.TIME})

        # Initialise scaler caches
        self._real_scalers = None
        self._target_scaler = None

        data = df[real_inputs].values
        targets = df[target_column].values
        self._real_scalers = sklearn.preprocessing.StandardScaler().fit(data)

        self._target_scaler = sklearn.preprocessing.StandardScaler().fit(targets)

        num_classes = {}
        self._num_classes_per_cat_input = num_classes

    def transform_inputs(self, df):
        """Performs feature transformations.

        This includes both feature engineering, preprocessing and normalisation.

        Args:
          df: Data frame to transform.

        Returns:
          Transformed data frame.

        """

        if self._real_scalers is None and self._cat_scalers is None:
            raise ValueError('Scalers have not been set!')

        # Extract relevant columns
        column_definitions = self.get_column_definition()

        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        df_res = df.copy()

        df_res[real_inputs] = self._real_scalers.transform(df_res[real_inputs].values)

        return df_res

    def format_predictions(self, predictions):
        """Reverts any normalisation to give predictions in original scale.

        Args:
          predictions: Dataframe of model predictions.

        Returns:
          Data frame of unnormalised predictions.
        """

        if self._target_scaler is None:
            raise ValueError('Scalers have not been set!')


        target_scaler = self._target_scaler
        target_columns = utils.extract_cols_from_input_type(InputTypes.TARGET, self._column_definition, {})

        predictions[target_columns] = target_scaler.inverse_transform(predictions[target_columns])


        return predictions

    def get_num_samples_for_calibration(self):
        """Gets the default number of training and validation samples.

        Use to sub-sample the data for network calibration and a value of -1 uses
        all available samples.

        Returns:
          Tuple of (training samples, validation samples)
        """
        return 450000, 50000
