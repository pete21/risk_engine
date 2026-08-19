from os import access, listdir, path, R_OK
from yaml import safe_load

from models.scoring_card import api_pb2

MODEL_PARAMETERS_LIST = ["scoring_card_name"]
MODEL_INPUT_SET = set()
DATA_PATH = path.join(path.dirname(__file__), 'data/')
SEP = '&'

for filename in listdir(DATA_PATH):
    if filename.endswith(".yaml"):
        file_path = path.join(DATA_PATH, filename)
        if path.exists(file_path) and access(file_path, R_OK):
            with open(file_path, 'r') as infile:
                try:
                    scoring_card_raw = safe_load(infile)
                except ValueError:
                    print('File {} is not a valid YAML file.'.format(file_path))
            for var_name in scoring_card_raw:
                if SEP in var_name:
                    MODEL_INPUT_SET.update(var_name.split(SEP))
                else:
                    MODEL_INPUT_SET.add(var_name)


def convert_to_model_parameters(dict_):
    return api_pb2.ModelParams(
        scoring_card_name=[v for k, v in dict_.items()
                           if k in MODEL_PARAMETERS_LIST][0]).SerializeToString()


def convert_to_model_input(dict_):
    return api_pb2.ModelInput(
        attributes_list=[api_pb2.ModelInput.Pair(
            key=k, value=v) for k, v in dict_.items() if k in MODEL_INPUT_SET]).SerializeToString()
