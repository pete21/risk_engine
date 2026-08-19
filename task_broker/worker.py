import argparse
import time

from task_broker.bpm_rest_client import RestClient, RestClientException
from task_broker.grpc_client import GrpcClient, GrpcClientException

from task_broker.grpc_converter import convert_to_model_parameters, convert_to_model_input


def run(bpm_server_url, topic_name, polling_rate, grpc_host, grpc_port, thread_id=0):

    bpm_client = RestClient(bpm_server_url)
    bpm_client.prepare_default_request_bodies(thread_id, topic_name)

    risk_engine_client = GrpcClient(grpc_host, grpc_port)

    delay = 1.0 / abs(polling_rate)

    msg_prefix = ""
    if thread_id is not None:
        msg_prefix = "Worker {}: ".format(thread_id)

    while True:
        try:
            status_code, res_json = bpm_client.fetch_and_lock()
        except RestClientException as err:

            print(msg_prefix + "Error calling fetch and lock ({}).".format(err))

            time.sleep(delay)
            continue

        print(msg_prefix+"Fetch and lock status code is {}, response is {}.".
              format(status_code, res_json))

        if len(res_json) != 0:
            id_ = res_json[0]["id"]
            model_name = res_json[0]["variables"].get("model_name", {"value": None})["value"]

            if model_name is None:
                print(msg_prefix+"No model_name in fetch and lock results - "
                                 "completing without calling model.")

                bpm_client.set_output_variables({"failure": "model_name not set"})
                status_code, res_json = bpm_client.complete(id_)

                print(msg_prefix+"Complete status code is {}, response is {}.".
                      format(status_code, res_json))

                time.sleep(delay)
                continue

            model_dict = {k: v["value"] for k, v in res_json[0]["variables"].items()}

            try:
                result_dict = risk_engine_client.send_request(
                    model_name,
                    model_parameters=convert_to_model_parameters[model_name](model_dict),
                    model_input=convert_to_model_input[model_name](model_dict),
                    msg_prefix=msg_prefix
                )
            except (GrpcClientException, KeyError) as err:
                result_dict = {"failure": "model calculation failed"}
                output_variable = "failure"

                print(msg_prefix + "Error calling model {} ({}).".format(model_name, err))
            else:
                output_variable = "result"

                print(msg_prefix + "result={}".format(result_dict["result"]))

            bpm_client.set_output_variables(
                {output_variable: result_dict[output_variable]})
            
            try:
                status_code, res_json = bpm_client.complete(id_)
            except RestClientException as err:

                print(msg_prefix + "Error calling complete ({}).".format(err))

                time.sleep(delay)
                continue

            print(msg_prefix+"Complete status code is {}, response is {}.".
                  format(status_code, res_json))

        time.sleep(delay)


def main():

    parser = argparse.ArgumentParser(description="Start BPM logistic regression worker.")
    parser.add_argument(
        "topic_name",
        type=str,
        help="topic name"
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

    run(args.server_url, args.topic_name, args.polling_rate, args.host, args.port)


if __name__ == "__main__":
    main()
