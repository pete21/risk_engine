import argparse
import time

import grpc

import risk_pb2
import risk_pb2_grpc

from models.avg import api_pb2 as avg_api_pb2
from models.sum import api_pb2 as sum_api_pb2
from models.logistic_regression import api_pb2 as logistic_regression_api_pb2
from models.risk_based_pricing import api_pb2 as risk_based_pricing_api_pb2


NEXT_CALL_ID = 0


def send_request(stub, model_name, **kw):
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
    send_request(
        stub, 'sum',
        model_input=sum_api_pb2.ModelInput(
            values=range(10)
        ).SerializeToString())

    # Use avg model
    send_request(
        stub, 'avg',
        model_parameters=avg_api_pb2.ModelParams(
            type=avg_api_pb2.ModelParams.ARITHMETIC
        ).SerializeToString(),
        model_input=avg_api_pb2.ModelInput(
            values=range(10)
        ).SerializeToString())

    send_request(
        stub, 'avg',
        model_parameters=avg_api_pb2.ModelParams(
            type=avg_api_pb2.ModelParams.GEOMETRIC
        ).SerializeToString(),
        model_input=avg_api_pb2.ModelInput(
            values=range(1, 10)
        ).SerializeToString())

    # Use logistic regression model
    send_request(
        stub, 'logistic_regression',
        model_input=logistic_regression_api_pb2.ModelInput(
            risk_attributes_list=[
                logistic_regression_api_pb2.ModelInput.Pair(
                    key='bank_account', value='nie'),
                logistic_regression_api_pb2.ModelInput.Pair(
                    key='age_in_years', value='33'),
                logistic_regression_api_pb2.ModelInput.Pair(
                    key='savings', value='from_1000_to_5000'),
                logistic_regression_api_pb2.ModelInput.Pair(
                    key='purpose', value='business'),
                logistic_regression_api_pb2.ModelInput.Pair(
                    key='property', value='car_or_other'),
                logistic_regression_api_pb2.ModelInput.Pair(
                    key='credit_history', value='exist_no_delays'),
                logistic_regression_api_pb2.ModelInput.Pair(
                    key='duration_in_month', value='12'),
                logistic_regression_api_pb2.ModelInput.Pair(
                    key='present_employment_since', value='more_than_7_years'),
                logistic_regression_api_pb2.ModelInput.Pair(
                    key='housing', value='rent'),
            ]
        ).SerializeToString())

    # Use risk based pricing model
    send_request(
        stub, 'risk_based_pricing',
        model_input=risk_based_pricing_api_pb2.ModelInput(
            pd=0.5).SerializeToString())

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
