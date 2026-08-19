from os import access, path, R_OK
from yaml import safe_load

from models import utils

from . import api_pb2

# This model has custom model parameters
decode_parameters = utils.ProtoDecoder(api_pb2.ModelParams)

# This model accepts list of pairs as input data
decode_input = utils.ProtoDecoder(api_pb2.ModelInput)

SEP = '&'
DATA_PATH = path.join(path.dirname(__file__), 'data/')


def list_accumulator():
    accumulator = []

    def appender(element=None):
        if element is not None:
            accumulator.append(element)
        else:
            return accumulator
    return appender


def split_to_tuple(string, split_list_accumulator=None):
    if SEP in string:
        split = tuple(string.split(SEP))
        if split_list_accumulator is not None:
            split_list_accumulator(split)
        return split
    else:
        return string


def load_scoring_card(filename):
    """Load scoring card as nested dictionary.
    :param string filename: name of file that contains scoring table
    :rtype: (dict, list)
    :return: scoring card dictionary and list of variable tuples in dictionary index
    :raises ValueError: file is nonexistent or unreadable
    :raises ValueError: scoring card content is invalid
    """
    file_path = DATA_PATH + filename
    if path.exists(file_path) and access(file_path, R_OK):
        with open(file_path, 'r') as infile:
            try:
                scoring_card_raw = safe_load(infile)
            except ValueError:
                raise ValueError('File {} is not valid YAML file.'.format(file_path))
    else:
        raise ValueError('File {} does not exist or has no read access.'.format(file_path))

    collect_var_tuples = list_accumulator()

    scoring_card = {
        split_to_tuple(var_name, split_list_accumulator=collect_var_tuples): {
            split_to_tuple(cat_name): cat_value for cat_name, cat_value in var_dict.items()}
        for var_name, var_dict in scoring_card_raw.items()}

    return scoring_card, collect_var_tuples()


def calculate(parameters, data):
    """Calculate credit score basing on attributes and scoring table.

    :param parameters: model parameters
    :param data: model data
    :rtype dict
    :return: dictionary containing calculated credit score
    :raises ValueError: input data is composed of empty list
    :raises ValueError: input data contains invalid probability
        of default value
    """
    scoring_card_filename = parameters.scoring_card_name
    attributes_list = data.attributes_list

    if len(attributes_list) == 0:
        raise ValueError('Input data list is empty.')

    attributes_dict = {d.key: d.value for d in attributes_list}
    scoring_card, tuple_list = load_scoring_card(scoring_card_filename)

    score = 0

    for var_name, cat_name in attributes_dict.items():
        cat_dict = scoring_card.get(var_name, None)
        if cat_dict is not None:
            score += cat_dict.get(cat_name, 0)

    for var_tuple in tuple_list:
        reject = False
        cat_name_list = []

        for var in var_tuple:
            cat_name = attributes_dict.get(var, None)
            if cat_name is None:
                reject = True
                continue
            cat_name_list.append(cat_name)

        if not reject:
            cat_dict = scoring_card.get(var_tuple, None)
            if cat_dict is not None:
                score += cat_dict.get(tuple(cat_name_list), 0)

    return {'result': score}
