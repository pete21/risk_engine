import argparse
import time

import grpc

import risk_pb2
import risk_pb2_grpc

from models.avg import api_pb2 as avg_api_pb2
from models.div import api_pb2 as div_api_pb2
from models.iris import api_pb2 as iris_api_pb2
from models.sum import api_pb2 as sum_api_pb2
from models.option_price import api_pb2 as option_api_pb2


NEXT_CALL_ID = 0


def sendRequest(stub, model_name, **kw):
    global NEXT_CALL_ID

    call_id = NEXT_CALL_ID
    NEXT_CALL_ID += 1

    def process(future):
        for key, value in sorted(future.result().values.items()):
            print('[{}/{}] {}: {}'.format(call_id, model_name, key, value))

    req = risk_pb2.CalculationRequest(model_name=model_name, **kw)
    future = stub.Calculate.future(req)

    future.add_done_callback(process)
    print('[{}/{}] Request sent'.format(call_id, model_name))


def run(host, port):
    channel = grpc.insecure_channel('%s:%d' % (host, port))
    stub = risk_pb2_grpc.RiskEngineStub(channel)

    # Use sum model
    sendRequest(
        stub, 'sum',
        model_input=sum_api_pb2.ModelInput(
            values=range(10)
        ).SerializeToString())

    # Use div model
    sendRequest(
        stub, 'div',
        model_input=div_api_pb2.ModelInput(
            dividend=10, divisor=3
        ).SerializeToString())

    # Use avg model
    sendRequest(
        stub, 'avg',
        model_parameters=avg_api_pb2.ModelParams(
            type=avg_api_pb2.ModelParams.ARITHMETIC
        ).SerializeToString(),
        model_input=avg_api_pb2.ModelInput(
            values=range(10)
        ).SerializeToString())

    sendRequest(
        stub, 'avg',
        model_parameters=avg_api_pb2.ModelParams(
            type=avg_api_pb2.ModelParams.GEOMETRIC
        ).SerializeToString(),
        model_input=avg_api_pb2.ModelInput(
            values=range(1, 10)
        ).SerializeToString())

    sendRequest(
        stub, 'iris',
        model_input=iris_api_pb2.ModelInput(
            sepal_length=5.0,
            sepal_width=3.6,
            petal_length=1.3,
            petal_width=0.25
        ).SerializeToString())


    try:
        print('Sleeping 10s waiting for results')
        time.sleep(10)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', help='host name', default='localhost', type=str)
    parser.add_argument('--port', help='port number', default=50052, type=int)

    args = parser.parse_args()
    run(args.host, args.port)
