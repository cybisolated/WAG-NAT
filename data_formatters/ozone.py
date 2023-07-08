"""Custom formatting functions for Ozone dataset.

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


class OzoneFormatter(GenericDataFormatter):
    """Defines and formats data for the electricity dataset.

    Note that per-entity z-score normalization is used here, and is implemented
    across functions.

    Attributes:
      column_definition: Defines input and data type of column used in the
        experiment.
      identifiers: Entity identifiers used in experiments.
    """

    _column_definition = [
        # id input
        ('id', DataTypes.REAL_VALUED, InputTypes.ID),
        # time input
        ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.TIME),
        # target
        ('O3', DataTypes.REAL_VALUED, InputTypes.TARGET),
        # future known_input
        # use time info as real value
        # ('year', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('month', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('day', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('hour', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('day_of_week', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        # use time info as categorical value
        # ('month', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        # ('day', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        # ('hour', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        # ('day_of_week', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        # future unkown input
        ('PM2.5', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('PM10', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('SO2', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('NO2', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('CO', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('TEMP', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('PRES', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('DEWP', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('RAIN', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('WSPM', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('wd', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
        # static input
        ('station', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
    ]

    def __init__(self):
        """Initialises formatter."""

        self.identifiers = None
        self._real_scalers = None
        self._cat_scalers = None
        self._target_scaler = None
        self._num_classes_per_cat_input = None

    def split_data(self, df, L_in, valid_start=1167, test_start=1314):
        """Splits data frame into training-validation-test data frames.

        This also calibrates scaling object, and transforms data for each split.

        Args:
          df: Source data frame to split.
          valid_start: Starting year for validation data
          test_start: Starting year for test data

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
        print('Setting scalers with training data...')

        column_definitions = self.get_column_definition()
        id_column = utils.get_single_col_by_input_type(InputTypes.ID, column_definitions)
        target_column = utils.get_single_col_by_input_type(InputTypes.TARGET, column_definitions)

        # Format real scalers
        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions, {InputTypes.ID, InputTypes.TIME}
        )

        # Initialise scaler caches
        self._real_scalers = {}
        self._target_scaler = {}
        identifiers = []
        for identifier, sliced in df.groupby(id_column):
            data = sliced[real_inputs].values
            targets = sliced[[target_column]].values
            # z-score normalize
            # self._real_scalers[identifier] \
            #     = sklearn.preprocessing.StandardScaler().fit(data)

            # self._target_scaler[identifier] \
            #     = sklearn.preprocessing.StandardScaler().fit(targets)
            # identifiers.append(identifier)

            # robust normalize
            # self._real_scalers[identifier] \
            #     = sklearn.preprocessing.RobustScaler().fit(data)

            # self._target_scaler[identifier] \
            #     = sklearn.preprocessing.RobustScaler().fit(targets)
            # identifiers.append(identifier)

            # min-max normalize
            self._real_scalers[identifier] = sklearn.preprocessing.MinMaxScaler().fit(data)

            self._target_scaler[identifier] = sklearn.preprocessing.MinMaxScaler().fit(targets)

            identifiers.append(identifier)

        # Format categorical scalers
        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions, {InputTypes.ID, InputTypes.TIME}
        )

        categorical_scalers = {}
        num_classes = {}
        for col in categorical_inputs:
            # Set all to str so that we don't have mixed integer/string columns
            srs = df[col].apply(str)
            categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(srs.values)
            num_classes[col] = srs.nunique()

        # Set categorical scaler outputs
        self._cat_scalers = categorical_scalers
        self._num_classes_per_cat_input = num_classes

        # Extract identifiers in case required
        self.identifiers = identifiers

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
        id_col = utils.get_single_col_by_input_type(InputTypes.ID, column_definitions)
        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions, {InputTypes.ID, InputTypes.TIME}
        )
        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions, {InputTypes.ID, InputTypes.TIME}
        )

        # Transform real inputs per entity
        df_list = []
        for identifier, sliced in df.groupby(id_col):
            sliced_copy = sliced.copy()
            sliced_copy[real_inputs] = self._real_scalers[identifier].transform(sliced_copy[real_inputs].values)
            df_list.append(sliced_copy)

        output = pd.concat(df_list, axis=0)

        # Format categorical inputs
        for col in categorical_inputs:
            string_df = df[col].apply(str)
            output[col] = self._cat_scalers[col].transform(string_df)

        return output

    def format_predictions(self, predictions):
        """Reverts any normalisation to give predictions in original scale.

        Args:
          predictions: Dataframe of model predictions.

        Returns:
          Data frame of unnormalised predictions.
        """

        if self._target_scaler is None:
            raise ValueError('Scalers have not been set!')

        column_names = predictions.columns

        df_list = []
        for identifier, sliced in predictions.groupby('station'):
            sliced_copy = sliced.copy()
            target_scaler = self._target_scaler[identifier]

            for col in column_names:
                if col not in {'forecast_time', 'station'}:
                    sliced_copy[col] = target_scaler.inverse_transform(sliced_copy[[col]])
            df_list.append(sliced_copy)

        output = pd.concat(df_list, axis=0)

        return output
