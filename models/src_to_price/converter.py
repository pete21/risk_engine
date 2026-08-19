from models.src_to_price import api_pb2


MODEL_PARAMETERS_LIST = ["conversion_table_name"]
MODEL_INPUT_LIST = ["src"]


def convert_to_model_parameters(dict_):
    return api_pb2.ModelParams(
        conversion_table_name=[v for k, v in dict_.items()
                               if k in MODEL_PARAMETERS_LIST][0]).SerializeToString()


def convert_to_model_input(dict_):
    return api_pb2.ModelInput(
        src=[v for k, v in dict_.items() if k in MODEL_INPUT_LIST][0]).SerializeToString()
