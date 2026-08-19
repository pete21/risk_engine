from functools import reduce
from os import access, path, R_OK

import pandas as pd

from models import utils

from . import api_pb2

# This model has no custom model parameters
# decode_parameters = utils.ProtoDecoder(api_pb2.ModelParams)

# This model accepts list of doubles as input data
decode_input = utils.ProtoDecoder(api_pb2.ModelInput)

PD_TO_RISK_RATING_PATH = path.join(path.dirname(__file__), 'data/pd_to_risk_rating.csv')
PRICING_MODEL_PATH = path.join(path.dirname(__file__), 'data/pricing_model.csv')


def load_model_components(pd_to_risk_rating_path, pricing_model_path):
    """Load prelearned logistic regression model, create logistic regression object and initialise
    it using prelearned model, load woe bins parameters.
    :rtype: (LogisticRegression object, dict, list)
    :return: logistic regression model, woe bins dictionary and model attribute list
    :raises ValueError: any of model files are nonexistent or unreadable
    :raises ValueError: woe file content is invalid
    """

    if path.exists(pd_to_risk_rating_path) and access(pd_to_risk_rating_path, R_OK):
        try:
            pd_to_risk_rating_df = pd.read_csv(pd_to_risk_rating_path)
        except ValueError:
            raise ValueError('File {} is not valid csv file.'.format(pd_to_risk_rating_path))
    else:
        raise ValueError('File {} does not exist or has no read access.'.format(pd_to_risk_rating_path))

    if path.exists(pricing_model_path) and access(pricing_model_path, R_OK):
        try:
            pricing_model_df = pd.read_csv(pricing_model_path)
        except ValueError:
            raise ValueError('File {} is not valid csv file.'.format(pricing_model_path))
        pricing_model_dict = dict(pricing_model_df.iloc[0].items())
    else:
        raise ValueError('File {} does not exist or has no read access.'
                         .format(pricing_model_path))

    return pd_to_risk_rating_df, pricing_model_dict


def cost_baseline(pd_, pm_dict):
    """Calculate cost baseline basing on probability of default and
    fixed pricing model attributes.
    :param float pd_: probability of default
    :param dict pm_dict: dictionary with pricing model attributes
    :return: calculated cost baseline
    :rtype: float
    """
    lgd = pm_dict["loss given default"]
    cost_of_risk = (pm_dict["risk free rate"] + pd_ * lgd) / (1 - pd_ * lgd)
    return pm_dict["cost of funds"] + pm_dict["cost of operations"] + cost_of_risk


def risk_rating(pd_, pd_to_risk_rating_df):
    """Find risk rating category corresponding to pd_.

    :param float pd_: probability of default
    :param df pd_to_risk_rating_df: data frame containing pd_ to
        risk rating mapping
    :rtype: string
    :return: string representing risk rating category corresponding
        to pd_.
    """
    if pd_ == pd_to_risk_rating_df.iloc[:, -1].values[1]:
        derived_risk_rating = pd_to_risk_rating_df.iloc[:, -1].name
    else:
        derived_risk_rating = \
            pd_to_risk_rating_df.loc[:, (pd_to_risk_rating_df.loc[0] <= pd_) &
                                        (pd_to_risk_rating_df.loc[1] > pd_)].columns.values[0]
    return derived_risk_rating


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
    pd_ = data.pd

    if not 0 <= pd_ <= 1:
        raise ValueError('Input data contains invalid probability of default: {}.'.format(pd_))

    pd_to_rr_df, pm_dict = load_model_components(PD_TO_RISK_RATING_PATH, PRICING_MODEL_PATH)

    cost_prediction, risk_r = cost_baseline(pd_, pm_dict), risk_rating(pd_, pd_to_rr_df)

    return {'result': cost_prediction,
            'risk_rating_int': reduce(lambda x, y: 256*x + y, (ord(c) for c in risk_r), 0)}
