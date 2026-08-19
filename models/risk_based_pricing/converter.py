from models.risk_based_pricing import api_pb2


MODEL_PARAMETERS_LIST = []
MODEL_INPUT_LIST = ["pd"]


def convert_to_model_parameters(dict_):
    return None


def convert_to_model_input(dict_):
    return api_pb2.ModelInput(
        pd=[v for k, v in dict_.items() if k in MODEL_INPUT_LIST][0]).SerializeToString()
