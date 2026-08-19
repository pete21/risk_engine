import json
from os import access, path, R_OK

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from models import utils

from . import api_pb2
from .woebin_ply import woebin_ply

# This model has no custom model parameters
# decode_parameters = utils.ProtoDecoder(api_pb2.ModelParams)

# This model accepts list of doubles as input data
decode_input = utils.ProtoDecoder(api_pb2.ModelInput)

LR_MODEL_PATH = path.join(path.dirname(__file__), 'data/lr_model.json')
WOE_BINS_PATH = path.join(path.dirname(__file__), 'data/woe_bins.json')


def load_model_components(lr_model_path, woe_bins_path):
    """Load prelearned logistic regression model, create logistic regression object and initialise
    it using prelearned model, load woe bins parameters.
    :rtype: (LogisticRegression object, dict, list)
    :return: logistic regression model, woe bins dictionary and model attribute list
    :raises ValueError: any of model files are nonexistent or unreadable
    :raises ValueError: woe file content is invalid
    """
    if path.exists(lr_model_path) and access(lr_model_path, R_OK):
        with open(lr_model_path, 'r') as infile:
            attr = json.load(infile)
        lr_model = LogisticRegression()
        for k, v in attr.items():
            if isinstance(v, list):
                setattr(lr_model, k, np.array(v))
            else:
                setattr(lr_model, k, v)
    else:
        raise ValueError('File {} does not exist or has no read access.'.format(lr_model_path))

    if path.exists(woe_bins_path) and access(woe_bins_path, R_OK):
        with open(woe_bins_path, 'r') as infile:
            woe_bins = json.load(infile)
        for k, v in woe_bins.items():
            try:
                woe_bins[k] = pd.read_json(v)
            except ValueError:
                raise ValueError('File {} does not encode valid data frames.'.format(woe_bins_path))
    else:
        raise ValueError('File {} does not exist or has no read access.'.format(woe_bins_path))

    return lr_model, woe_bins, [*woe_bins]


def calculate(parameters, data):
    """Calculate probabilities of default for a list of risk attribute dictionaries using logistic
    regression on prelearned model.

    :param parameters: model parameters
    :param data: model data
    :rtype list of floats
    :return: list of calculated probabilities of default
    :raises ValueError: input data is composed of empty list
    :raises ValueError: input data contains invalid attribute name
    """
    ra_struct_list = data.risk_attributes_list

    if len(ra_struct_list) == 0:
        raise ValueError('Input data list is empty.')

    risk_attribute_dict = {d.key: d.value for d in ra_struct_list}

    lr, bins, attribute_list = load_model_components(LR_MODEL_PATH, WOE_BINS_PATH)

    print(attribute_list)

    for attribute in risk_attribute_dict:
        if attribute not in attribute_list:
            raise ValueError('Input data contains at least one invalid attribute name: {}.'
                             .format(attribute))

    for attribute in attribute_list:
        if attribute in ["age_in_years", "duration_in_month"] and attribute in risk_attribute_dict:
            risk_attribute_dict[attribute] = int(risk_attribute_dict[attribute])
        if attribute not in risk_attribute_dict:
            risk_attribute_dict[attribute] = "missing"

    risk_attributes_df = pd.DataFrame([risk_attribute_dict])
    risk_attributes_woe = woebin_ply(risk_attributes_df, bins)
    risk_attributes_woe.fillna(0, inplace=True)

    pd_predictions = lr.predict_proba(risk_attributes_woe)[:, 1]

    return {'result': pd_predictions[0]}
