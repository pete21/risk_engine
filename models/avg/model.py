from models import utils

from . import api_pb2

# This model has custom model parameters
decode_parameters = utils.ProtoDecoder(api_pb2.ModelParams)

# This model accepts list of doubles as input data
decode_input = utils.ProtoDecoder(api_pb2.ModelInput)

def calculate(parameters, data):
    count = len(data.values)
    if not count:
        raise ValueError('At least one number required to calculate average')
    if parameters.type == api_pb2.ModelParams.ARITHMETIC:
        total = sum(data.values)
        return {
            'total': total,
            'count': count,
            'result': total / count
        }
    elif parameters.type == api_pb2.ModelParams.GEOMETRIC:
        total = 1.0
        for x in data.values:
            total *= x
        return {
            'total': total,
            'count': count,
            'result': pow(total, 1 / count)
        }
    else:
        raise ValueError('Unknown average type: ' + parameters.type)
