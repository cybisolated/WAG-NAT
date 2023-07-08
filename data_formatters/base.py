"""Default data formatting functions for experiments.

For new datasets, inherit form GenericDataFormatter and implement
all abstract functions.

These dataset-specific methods:
1) Define the column and input types for tabular dataframes used by model
2) Perform the necessary input feature engineering & normalisation steps
3) Reverts the normalisation for predictions
4) Are responsible for train, validation and test splits


"""

import abc
import enum


# Type defintions
class DataTypes(enum.IntEnum):
    """Defines numerical types of each column."""
    REAL_VALUED = 0
    CATEGORICAL = 1
    DATE = 2


class InputTypes(enum.IntEnum):
    """Defines input types of each column."""
    TARGET = 0
    OBSERVED_INPUT = 1
    KNOWN_INPUT = 2
    STATIC_INPUT = 3
    ID = 4  # Single column used as an entity identifier
    TIME = 5  # Single column exclusively used as a time index


class GenericDataFormatter(abc.ABC):
    """Abstract base class for all data formatters.

    User can implement the abstract methods below to perform dataset-specific
    manipulations.

    """

    @abc.abstractmethod
    def set_scalers(self, df):
        """Calibrates scalers using the data supplied."""
        raise NotImplementedError()

    @abc.abstractmethod
    def transform_inputs(self, df):
        """Performs feature transformation."""
        raise NotImplementedError()

    @abc.abstractmethod
    def format_predictions(self, df):
        """Reverts any normalisation to give predictions in original scale."""
        raise NotImplementedError()

    @abc.abstractmethod
    def split_data(self, df):
        """Performs the default train, validation and test splits."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def _column_definition(self):
        """Defines order, input type and data type of each column."""
        raise NotImplementedError()

    # Shared functions across data-formatters

    @property
    def num_classes_per_cat_input(self):
        """Returns number of categories per relevant input.

        This is seqeuently required for keras embedding layers.
        """
        return self._num_classes_per_cat_input

    def get_column_definition(self):
        """"Returns formatted column definition in order expected by the TFT."""

        column_definition = self._column_definition

        # Sanity checks first.
        # Ensure only one ID and time column exist
        def _check_single_column(input_type):

            length = len([tup for tup in column_definition if tup[2] == input_type])

            if length != 1:
                raise ValueError('Illegal number of inputs ({}) of type {}'.format(
                    length, input_type))

        _check_single_column(InputTypes.ID)
        _check_single_column(InputTypes.TIME)

        identifier = [tup for tup in column_definition if tup[2] == InputTypes.ID]
        time = [tup for tup in column_definition if tup[2] == InputTypes.TIME]
        real_inputs = [
            tup for tup in column_definition if tup[1] == DataTypes.REAL_VALUED and
            tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]
        categorical_inputs = [
            tup for tup in column_definition if tup[1] == DataTypes.CATEGORICAL and
            tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]

        return identifier + time + real_inputs + categorical_inputs

    def get_input_columns(self):
        """Returns names of all input columns."""
        return [
            tup[0]
            for tup in self.get_column_definition()
            if tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]

    def get_input_names(self):
        def _get_names(input_types, data_type, defn):
            return [tup[0] for i, tup in enumerate(defn) if tup[1] == data_type and tup[2] in input_types]

        column_definition = [
            tup for tup in self.get_column_definition()
            if tup[2] not in {InputTypes.TIME}
        ]

        return dict(
            group_id=_get_names({InputTypes.ID}, DataTypes.REAL_VALUED, column_definition),
            target=_get_names({InputTypes.TARGET}, DataTypes.REAL_VALUED, column_definition),
            time_varying_known_reals=_get_names({InputTypes.KNOWN_INPUT}, DataTypes.REAL_VALUED, column_definition),
            time_varying_known_categoricals=_get_names(
                {InputTypes.KNOWN_INPUT}, DataTypes.CATEGORICAL, column_definition),
            time_varying_unknown_reals=_get_names({InputTypes.OBSERVED_INPUT},
                                                  DataTypes.REAL_VALUED, column_definition),
            time_varying_unknown_categoricals=_get_names(
                {InputTypes.OBSERVED_INPUT}, DataTypes.CATEGORICAL, column_definition),
            static_reals=_get_names({InputTypes.STATIC_INPUT}, DataTypes.REAL_VALUED, column_definition),
            static_categoricals=_get_names({InputTypes.STATIC_INPUT}, DataTypes.CATEGORICAL, column_definition)
        )

    def get_input_indices(self):
        """Returns the relevant indexes and input sizes required by model."""

        # Functions
        def _extract_tuples_from_data_type(data_type, defn):
            return [
                tup for tup in defn if tup[1] == data_type and
                tup[2] not in {InputTypes.ID, InputTypes.TIME}
            ]

        def _get_locations_from_input_type(input_types, defn):
            return [i for i, tup in enumerate(defn) if tup[2] in input_types]

        def _get_locations(input_types, data_type, defn):
            return [i for i, tup in enumerate(defn) if tup[1] == data_type and tup[2] in input_types]

        def _get_num_classes(defn):
            return {i: self.num_classes_per_cat_input[tup[0]] for i, tup in enumerate(defn)
                    if tup[1] == DataTypes.CATEGORICAL}

        # Start extraction
        column_definition = [
            tup for tup in self.get_column_definition()
            if tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]

        # categorical_inputs = _extract_tuples_from_data_type(DataTypes.CATEGORICAL,
        #                                                     column_definition)
        # real_inputs = _extract_tuples_from_data_type(DataTypes.REAL_VALUED,
        #                                              column_definition)

        locations = dict(
            input_size=len(self.get_input_columns()),
            output_size=len(_get_locations_from_input_type({InputTypes.TARGET}, column_definition)),
            num_classes=_get_num_classes(column_definition),
            target_loc=_get_locations_from_input_type({InputTypes.TARGET}, column_definition),
            time_varying_knwon_real_inputs_loc=_get_locations(
                {InputTypes.KNOWN_INPUT}, DataTypes.REAL_VALUED, column_definition),
            time_varying_knwon_categorical_inputs_loc=_get_locations(
                {InputTypes.KNOWN_INPUT}, DataTypes.CATEGORICAL, column_definition),
            time_varying_unknwon_real_inputs_loc=_get_locations({InputTypes.OBSERVED_INPUT},
                                                                DataTypes.REAL_VALUED, column_definition),
            time_varying_unknwon_categorical_inputs_loc=_get_locations(
                {InputTypes.OBSERVED_INPUT}, DataTypes.CATEGORICAL, column_definition),
            static_inputs=_get_locations_from_input_type({InputTypes.STATIC_INPUT}, column_definition)
        )

        return locations
