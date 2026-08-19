from task_broker.bpm_rest_client import RestClientException
from task_broker.grpc_client import GrpcClientException

from task_broker.grpc_converter import convert_to_model_parameters, convert_to_model_input


def run(bpm_client, risk_engine_client, recv_event, thread_id, idle_workers_queue):

    msg_prefix = "Worker {}: ".format(thread_id)

    while True:
        idle_workers_queue.append(thread_id)

        recv_event.wait()
        recv_event.clear()

        res_json = bpm_client.get_last_fetch_and_lock_response()
        id_ = res_json[0]["id"]
        model_name = res_json[0]["variables"].get("model_name", {"value": None})["value"]

        if model_name is None:
            print(msg_prefix+"No model_name in fetch and lock results - "
                             "completing without calling model.")

            bpm_client.set_output_variables({"failure": "model_name not set"})
            status_code, res_json = bpm_client.complete(id_)

            print(msg_prefix+"Complete status code is {}, response is {}.".
                  format(status_code, res_json))

            continue

        model_dict = {k: v["value"] for k, v in res_json[0]["variables"].items()}
        variable_dict = {}

        try:
            result_dict = risk_engine_client.send_request(
                model_name,
                model_parameters=convert_to_model_parameters[model_name](model_dict),
                model_input=convert_to_model_input[model_name](model_dict),
                msg_prefix=msg_prefix
            )
        except (GrpcClientException, KeyError) as err:
            variable_dict["error_code"] = 1

            print(msg_prefix + "Error calling model {} ({}).".format(model_name, err))

        else:
            variable_dict["error_code"] = 0
            variable_dict["result"] = result_dict["result"]

            print(msg_prefix + "result={}".format(result_dict["result"]))

        bpm_client.set_output_variables(variable_dict)

        try:
            status_code, res_json = bpm_client.complete(id_)
        except RestClientException as err:

            print(msg_prefix + "Error calling complete ({}).".format(err))

            continue

        print(msg_prefix+"Complete status code is {}, response is {}.".
              format(status_code, res_json))
