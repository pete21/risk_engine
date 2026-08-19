from os import access, path, R_OK

import pandas as pd

from models import utils

from . import api_pb2

# This model has custom model parameter
decode_parameters = utils.ProtoDecoder(api_pb2.ModelParams)

# This model accepts integer as input datum
decode_input = utils.ProtoDecoder(api_pb2.ModelInput)

DATA_PATH = path.join(path.dirname(__file__), 'data/')


def load_reference_scale(filename):
    """Load reference scale as pandas dataframe.
    :param string filename: name of file that contains reference scale
    :rtype: df
    :return: reference scale dataframe
    :raises ValueError: file is nonexistent or unreadable
    :raises ValueError: scoring card content is invalid
    """
    file_path = DATA_PATH + filename

    if path.exists(file_path) and access(file_path, R_OK):
        try:
            reference_scale_df = pd.read_csv(
                file_path, dtype={"lower bound": int, "pd": float, "src": float})
        except ValueError:
            raise ValueError(
                'File {} is not valid CSV file or contains ill-formed data.'.format(file_path))
    else:
        raise ValueError('File {} does not exist or has no read access.'.format(file_path))

    if list(reference_scale_df.columns.values) != ['lower bound', 'pd', 'src']:
        raise ValueError('CSV file {} contains wrong set of columns.'.format(file_path))

    if reference_scale_df.isnull().values.any():
        raise ValueError('Some values are missing in CSV file {}.'.format(file_path))

    return reference_scale_df


def apply_scale(score, reference_scale_df):
    """Find PD and SRC values corresponding to score according to reference scale.

    :param float score: credit score
    :param df reference_scale_df: dataframe containing score to PD and
        SRC mapping
    :rtype: (float, float)
    :return: PD and SRC values
    """
    refscale_rows_below = reference_scale_df[reference_scale_df["lower bound"] <= score]
    max_bound_value_series = refscale_rows_below.loc[refscale_rows_below["lower bound"].idxmax()]
    return float(max_bound_value_series["pd"]), float(max_bound_value_series["src"])


def calculate(parameters, data):
    """Calculate cost predictions and risk ratings from probabilities
    of default basing on fixed pricing model attributes and risk
    ratting mapping.

    :param parameters: model parameters
    :param data: model data
    :rtype list of floats
    :return: list of calculated credit costs
    :raises ValueError: input data is composed of empty list
    :raises ValueError: input data contains invalid probability
        of default value
    """
    reference_scale_filename = parameters.reference_scale_name
    score = data.score

    reference_scale_df = load_reference_scale(reference_scale_filename)

    pd_, src = apply_scale(score, reference_scale_df)

    return {'result': src,
            'probability_of_default': pd_}
