from models.logistic_regression import api_pb2

MODEL_PARAMETERS_LIST = []
MODEL_INPUT_LIST = ["bank_account", "duration_in_month", "credit_history", "purpose", "savings",
                    "present_employment_since", "property", "age_in_years", "housing"]


def convert_to_model_parameters(dict_):
    return None


def convert_to_model_input(dict_):
    return api_pb2.ModelInput(
        risk_attributes_list=[api_pb2.ModelInput.Pair(
            key=k, value=v) for k, v in dict_.items() if k in MODEL_INPUT_LIST]).SerializeToString()
