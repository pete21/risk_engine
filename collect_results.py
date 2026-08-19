import argparse
import json

from os import path
from uuid import uuid1

from task_broker.bpm_rest_client import RestClient, RestClientException


def get_variables(bpm_client, process_instance_id):

    bpm_client.prepare_default_request_bodies(uuid1().hex, process_instance_id)

    try:
        status_code, res_json = bpm_client.fetch_and_lock()
    except RestClientException as err:

        print("Error calling fetch and lock ({}).".format(err))

        return None, None

    return status_code, res_json


def end_process(bpm_client, worker_id):

    try:
        status_code, res_json = bpm_client.complete(worker_id)
    except RestClientException as err:

        print("Error calling complete ({}).".format(err))

        return None, None

    return status_code, res_json


def write_dict_to_file(dict_, file_path, overwrite=False):
    if not overwrite and path.exists(file_path):
        raise FileExistsError
    else:
        with open(file_path, 'w') as outfile:
            json.dump(dict_, outfile, sort_keys=True, indent=2)


class ValidateIdAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) <= 20:
            print("Id {} is too short and can conflict with topic names used in BPMN diagram."
                  .format(values))
            raise ValueError("Invalid id")
        setattr(namespace, self.dest, values)


def main():

    parser = argparse.ArgumentParser(
        description="Query Camunda server for a process instance in its final state. "
                    "If found collect process instance variables and finalize process instance, "
                    "exit otherwise.")
    parser.add_argument(
        "process_instance_id",
        type=str,
        action=ValidateIdAction,
        help="Id of process instance to be queried."
    )
    parser.add_argument(
        "--server_url",
        type=str,
        help="set BPM server URL to override default",
        default="http://localhost:8080"
    )
    parser.add_argument(
        "--json_file_path",
        type=str,
        help="file path to save JSON file containing variable values",
        default=None
    )
    parser.add_argument(
        "--overwrite",
        action='store_true',
        help="OK to overwrite existing file",
    )

    try:
        args = parser.parse_args()
    except ValueError as error_code:
        print("Argument parser raised ValueError: {}, exiting.".format(error_code))
        raise SystemExit(1)

    bpm_client = RestClient(args.server_url)

    status_code, res_json = get_variables(bpm_client, args.process_instance_id)

    print("Fetch and lock status code is {}.".format(status_code))

    if res_json:
        id_ = res_json[0]["id"]
        variable_dict = {k: v["value"] for k, v in res_json[0]["variables"].items()}

        if args.json_file_path is not None:
            try:
                write_dict_to_file(res_json[0]["variables"], args.json_file_path, args.overwrite)
            except OSError as error_code:

                print("File {} already exists or cannot be created, variable values not saved. {}"
                      .format(args.json_file_path, error_code))
                print("Dumping variable dictionary to screen instead:")
                print(json.dumps(res_json[0]["variables"], sort_keys=True, indent=2))

            else:

                print("Variable values saved to file {}.".format(args.json_file_path))

        else:
            print("Process variables:")
            print(json.dumps(variable_dict, sort_keys=True, indent=2))

        status_code, _ = end_process(bpm_client, id_)

        print("Complete status code is {}.".format(status_code))

        error_code = variable_dict.get("error_code", None)

        if error_code is not None:

            print("Process error code is {}.".format(error_code))

            if not error_code:
                final_result = variable_dict.get("final_result", None)
                if final_result is not None:

                    print("Process final result is {}.".format(final_result))

    else:

        print("Fetch and lock result array is empty. Check if process instance id {} is correct "
              "or wait for the process instance to finish and execute {} script again."
              .format(args.process_instance_id, parser.prog))


if __name__ == "__main__":
    main()
