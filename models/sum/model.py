import math

from models import utils

from . import api_pb2

# This model accepts list of doubles as input data
decode_input = utils.ProtoDecoder(api_pb2.ModelInput)

def calculate(parameters, data):
    return {'result': math.fsum(data.values)}
