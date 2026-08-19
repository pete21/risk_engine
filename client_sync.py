import argparse

import grpc

import risk_pb2
import risk_pb2_grpc

from models.avg import api_pb2 as avg_api_pb2
from models.div import api_pb2 as div_api_pb2
from models.sum import api_pb2 as sum_api_pb2


def printResponse(stub, req):
    print('Sending', req.model_name, 'request')
    res = stub.Calculate(req)
    for key, value in sorted(res.values.items()):
        print('{}: {}'.format(key, value))
    print()


def run(host, port):
    channel = grpc.insecure_channel('%s:%d' % (host, port))
    stub = risk_pb2_grpc.RiskEngineStub(channel)

    # Use sum model
    req = risk_pb2.CalculationRequest(
        model_name='sum',
        model_input=sum_api_pb2.ModelInput(values=range(10)).SerializeToString())
    printResponse(stub, req)

    # Use div model
    req = risk_pb2.CalculationRequest(
        model_name='div',
        model_input=div_api_pb2.ModelInput(dividend=10, divisor=3).SerializeToString())
    printResponse(stub, req)

    # Use avg model
    req = risk_pb2.CalculationRequest(
        model_name='avg',
        model_parameters=avg_api_pb2.ModelParams(
            type=avg_api_pb2.ModelParams.ARITHMETIC
        ).SerializeToString(),
        model_input=avg_api_pb2.ModelInput(values=range(10)).SerializeToString())
    printResponse(stub, req)

    req = risk_pb2.CalculationRequest(
        model_name='avg',
        model_parameters=avg_api_pb2.ModelParams(
            type=avg_api_pb2.ModelParams.GEOMETRIC
        ).SerializeToString(),
        model_input=avg_api_pb2.ModelInput(values=range(1, 10)).SerializeToString())
    printResponse(stub, req)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', help='host name', default='localhost', type=str)
    parser.add_argument('--port', help='port number', default=50052, type=int)

    args = parser.parse_args()
    run(args.host, args.port)
