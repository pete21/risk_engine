import argparse
from collections import namedtuple
import time

from task_broker import bpm_rest_client
from models.logistic_regression.model import calculate, load_model_components,\
    LR_MODEL_PATH, WOE_BINS_PATH

Data = namedtuple('Data', ['risk_attributes_list'])
Dictionary = namedtuple('Dictionary', ['pairs'])
Pair = namedtuple('Pair', ['key', 'value'])


def convert_to_protobuf_style(dict):
    return Data(risk_attributes_list=[Dictionary(pairs=[Pair(key=k, value=v)
                                                        for k, v in dict.items()])])


def run(server_url, worker_id, topic_name, output_variable, polling_rate):

    if server_url is None:
        client = bpm_rest_client.RestClient()
    else:
        client = bpm_rest_client.RestClient(server_url)

    if polling_rate is None:
        delay = 1.0
    else:
        delay = 1.0 / abs(polling_rate)

    _, _, task_input_variables = load_model_components(LR_MODEL_PATH, WOE_BINS_PATH)

    client.prepare_default_request_bodies(worker_id, topic_name, task_input_variables)

    while True:
        status_code, res_json = client.fetch_and_lock()

        print("Fetch and lock status code is {}, response is {}.".format(status_code, res_json))

        if len(res_json) != 0:
            model_variables_dict = {k: v["value"] for k, v in res_json[0]["variables"].items()}

            pd_dict = calculate(None, convert_to_protobuf_style(model_variables_dict))

            print("PD={}".format(pd_dict['result']))

            id_ = res_json[0]["id"]
            client.set_output_variables({output_variable: pd_dict['result']})
            status_code, res_json = client.complete(id_)

            print("Complete status code is {}, response is {}.".format(status_code, res_json))

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
        help="set server URL to override default"
    )
    parser.add_argument(
        "--polling-rate",
        type=float,
        help="set BPM polling rate in Hz to override default"
    )

    args = parser.parse_args()

    run(args.server_url, args.worker_id, args.topic_name, args.output_variable, args.polling_rate)


if __name__ == "__main__":
    main()
