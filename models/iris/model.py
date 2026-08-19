import os
from sklearn.externals import joblib

from models import utils

from . import api_pb2

_MODEL = joblib.load(
    os.path.join(os.path.dirname(__file__), 'iris_model.pickle'))

decode_input = utils.ProtoDecoder(api_pb2.ModelInput)

def calculate(parameters, data):
    result = _MODEL.predict([[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
    ]])
    return {'species': result[0]}
