import argparse
import json

from task_broker.bpm_rest_client import RestClient, RestClientException


class TriggerModelException(Exception):
    pass


def get_model_key_list(bpm_client):

    try:
        status_code, res_json = bpm_client.get_process_definitions()
    except RestClientException as err:

        print("Error getting process definitions ({}).".format(err))

    else:

        if res_json:
            return [dict_["key"] for dict_ in res_json]

        return []


def load_variables(filename):

    if filename is None:
        return {}

    try:
        with open(filename, 'r') as infile:
            variables_dict = json.load(infile)
    except IOError as err:
        raise TriggerModelException("Cannot open variables_json file: {}.".format(err)) from None

    return variables_dict


def start_process(bpm_client, args, variables_dict):

    try:
        status_code, res_json = bpm_client.start_process_instance(
            args.definition_key, args.business_key, variables_dict)
    except RestClientException as err:
        raise TriggerModelException("Error starting process instance ({}).".format(err)) from None

    else:
        return status_code, res_json["id"]


def main():

    parser = argparse.ArgumentParser(
        description="Start process for given BPMN diagram and pass to it variables defined in "
                    "a JSON file . List all BPMN diagrams deployed on Camunda server if diagram "
                    "key is not provided - ignore the rest of command line arguments in this case.")
    parser.add_argument(
        "--definition_key",
        type=str,
        help="definition key",
        default=None
    )
    parser.add_argument(
        "--server_url",
        type=str,
        help="set BPM server URL to override default",
        default="http://localhost:8080"
    )
    parser.add_argument(
        "--variables_json",
        type=str,
        help="file path to JSON file containing variable definitions",
        default=None
    )
    parser.add_argument(
        "--business_key",
        type=str,
        help="set business key to override default",
        default=""
    )

    args = parser.parse_args()

    bpm_client = RestClient(args.server_url)

    if args.definition_key is not None:
        variables_dict = load_variables(args.variables_json)
        status_code, process_id = start_process(bpm_client, args, variables_dict)

        print("Start process instance status code is {}, process instance id is {}.".
              format(status_code, process_id))

    else:
        model_key_list = get_model_key_list(bpm_client)

        if model_key_list:

            print("Model list: {}.".format(model_key_list))

        else:

            print("Model list is empty.")


if __name__ == "__main__":
    main()
