import math

from models import utils

from . import api_pb2

# This model accepts exactly two doubles as input data
decode_input = utils.ProtoDecoder(api_pb2.ModelInput)

def calculate(parameters, data):
    remainder = math.fmod(data.dividend, data.divisor)
    return {
        'result': data.dividend / data.divisor,
        'quotient': (data.dividend - remainder) / data.divisor,
        'remainder': remainder,
    }
