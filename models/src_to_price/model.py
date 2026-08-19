from os import access, path, R_OK

import pandas as pd

from models import utils

from . import api_pb2

# This model has custom model parameter
decode_parameters = utils.ProtoDecoder(api_pb2.ModelParams)

# This model accepts integer as input datum
decode_input = utils.ProtoDecoder(api_pb2.ModelInput)

DATA_PATH = path.join(path.dirname(__file__), 'data/')


def load_table(filename):
    """Load csv table as pandas dataframe.
    :param string filename: name of file that contains reference scale
    :rtype: df
    :return: reference scale dataframe
    :raises ValueError: file is nonexistent or unreadable
    :raises ValueError: scoring card content is invalid
    """
    file_path = DATA_PATH + filename

    if path.exists(file_path) and access(file_path, R_OK):
        try:
            table_df = pd.read_csv(
                file_path, dtype={"src lower bound": int, "price": float})
        except ValueError:
            raise ValueError(
                'File {} is not valid CSV file or contains ill-formed data.'.format(file_path))
    else:
        raise ValueError('File {} does not exist or has no read access.'.format(file_path))

    if list(table_df.columns.values) != ['src lower bound', 'price']:
        raise ValueError('CSV file {} contains wrong set of columns.'.format(file_path))

    if table_df.isnull().values.any():
        raise ValueError('Some values are missing in CSV file {}.'.format(file_path))

    return table_df


def apply_scale(src, reference_scale_df):
    """Find price value corresponding to SRC according to reference scale.

    :param float src: SRC value
    :param df reference_scale_df: dataframe containing SRC to price
    :rtype: float
    :return: price value
    """
    refscale_rows_below = reference_scale_df[reference_scale_df["src lower bound"] <= src]
    max_bound_value_series = refscale_rows_below.loc[
        refscale_rows_below["src lower bound"].idxmax()]
    return float(max_bound_value_series["price"])


def calculate(parameters, data):
    """Calculate price from SRC value.

    :param parameters: model parameters
    :param data: model data
    :rtype dict
    :return: credit price
    """
    conversion_table_filename = parameters.conversion_table_name
    src = data.src

    conversion_table_df = load_table(conversion_table_filename)

    price = apply_scale(src, conversion_table_df)

    return {'result': price}
