import argparse
import time

import grpc

from task_broker import bpm_rest_client
from models.logistic_regression import api_pb2 as logistic_regression_api_pb2
import risk_pb2
import risk_pb2_grpc

ATTRIBUTE_LIST = ['bank_account', 'duration_in_month', 'credit_history', 'purpose', 'savings',
                  'present_employment_since', 'property', 'age_in_years', 'housing']


def send_request(stub, model_name, model_input, msg_prefix=""):
    req = risk_pb2.CalculationRequest(model_name=model_name, model_input=model_input)

    print(msg_prefix+'Sending {} request.'.format(model_name))

    res = stub.Calculate(req)

    for key, value in sorted(res.values.items()):
        print(msg_prefix+'{}: {}'.format(key, value))
    print()

    return res.values


def convert_to_protobuf_style(dict_):
    return logistic_regression_api_pb2.ModelInput(
        risk_attributes_list=[logistic_regression_api_pb2.ModelInput.Dictionary(
            pairs=[logistic_regression_api_pb2.ModelInput.Pair(
                key=k, value=v) for k, v in dict_.items()])]).SerializeToString()


def run(bpm_server_url, worker_id, topic_name, output_variable, polling_rate, grpc_host, grpc_port, thread_id=None):

    bpm_client = bpm_rest_client.RestClient(bpm_server_url)
    delay = 1.0 / abs(polling_rate)

    task_input_variables = ATTRIBUTE_LIST

    bpm_client.prepare_default_request_bodies(worker_id, topic_name, task_input_variables)

    channel = grpc.insecure_channel('%s:%d' % (grpc_host, grpc_port))
    stub = risk_pb2_grpc.RiskEngineStub(channel)

    msg_prefix = ""
    if thread_id is not None:
        msg_prefix = "Worker {}: ".format(thread_id)

    while True:
        status_code, res_json = bpm_client.fetch_and_lock()

        print(msg_prefix+"Fetch and lock status code is {}, response is {}.".format(status_code, res_json))

        if len(res_json) != 0:
            model_variables_dict = {k: str(v["value"]) for k, v in res_json[0]["variables"].items()}

            pd_dict = send_request(
                stub,
                'logistic_regression',
                convert_to_protobuf_style(model_variables_dict),
                msg_prefix
            )

            print(msg_prefix+"PD={}".format(pd_dict['result']))

            id_ = res_json[0]["id"]
            bpm_client.set_output_variables({output_variable: pd_dict['result']})
            status_code, res_json = bpm_client.complete(id_)

            print(msg_prefix+"Complete status code is {}, response is {}.".format(status_code, res_json))

        time.sleep(delay)


def main():

    parser = argparse.ArgumentParser(description="Start BPM logistic regression worker.")
    parser.add_argument(
        "worker_id",
        type=str,
        help="worker id"
    )
    parser.add_argument(
        "topic_name",
        type=str,
        help="topic name"
    )
    parser.add_argument(
        "output_variable",
        type=str,
        help="worker output variable name"
    )
    parser.add_argument(
        "--server-url",
        type=str,
        help="set BPM server URL to override default",
        default="http://localhost:8080"
    )
    parser.add_argument(
        "--polling-rate",
        type=float,
        help="set BPM polling rate in Hz to override default",
        default=1.0
    )
    parser.add_argument(
        '--host',
        type=str,
        help='set grpc host name to override default',
        default='localhost'
    )
    parser.add_argument(
        '--port',
        type=int,
        help='set grpc port number to override default',
        default=10000)

    args = parser.parse_args()

    run(args.server_url, args.worker_id, args.topic_name, args.output_variable,
        args.polling_rate, args.host, args.port)


if __name__ == "__main__":
    main()
