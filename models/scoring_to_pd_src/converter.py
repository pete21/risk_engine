from models.scoring_to_pd_src import api_pb2


MODEL_PARAMETERS_LIST = ["reference_scale_name"]
MODEL_INPUT_LIST = ["score"]


def convert_to_model_parameters(dict_):
    return api_pb2.ModelParams(
        reference_scale_name=[v for k, v in dict_.items()
                              if k in MODEL_PARAMETERS_LIST][0]).SerializeToString()


def convert_to_model_input(dict_):
    return api_pb2.ModelInput(
        score=[v for k, v in dict_.items() if k in MODEL_INPUT_LIST][0]).SerializeToString()
