import importlib
from os import listdir

directory_list = listdir("models")
model_dict = {}

for model_name in directory_list:
    full_name = "models." + model_name + ".converter"
    try:
        model_dict[model_name] = importlib.import_module(full_name)
    except ImportError as err:

        print("Import from directory {} failed. ({})".format(model_name, err))

    else:

        print("Module {} imported successfully!".format(full_name))

convert_to_model_parameters = {
    model_name: getattr(model_lib, "convert_to_model_parameters")
    for model_name, model_lib in model_dict.items()}


convert_to_model_input = {
    model_name: getattr(model_lib, "convert_to_model_input")
    for model_name, model_lib in model_dict.items()}
