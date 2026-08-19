import json
import requests

LOCAL_SERVER_DEFAULT_URL = "http://localhost:8080"
YEAR_IN_MILLISECONDS = 31556926000


class RestClient(object):
    """Client for Camunda REST API.
    """

    def __init__(self, server_url=LOCAL_SERVER_DEFAULT_URL):
        """Initialises attributes for API calls.
        :param str server_url: Camunda server URL
        """

        self._server_url = server_url
        self._default_fetch_and_lock_rb = None
        self._default_complete_rb = None
        self._last_res_json = None

    def prepare_default_request_bodies(
            self, worker_id, topic_name, max_tasks=1, lock_duration=YEAR_IN_MILLISECONDS,
            use_priority=True):

        self._default_fetch_and_lock_rb = {
            "workerId": worker_id,
            "maxTasks": max_tasks,
            "usePriority": "true" if use_priority else "false",
            "topics":
                [{"topicName": topic_name,
                  "lockDuration": lock_duration
                  }]
        }

        self._default_complete_rb = {
            "workerId": worker_id,
            "variables": {}
        }

    def set_output_variables(self, output_variables_dict):
        self._default_complete_rb["variables"] = {
            name: {"value": value} for name, value in output_variables_dict.items()
        }

    @staticmethod
    def _process_response(response):
        """Check response status code. Raise exception if status code
        indicates request failure, return decoded json response
        otherwise.
        :param response: response object
        :return: status code and decoded json response
        :rtype: (string, list or dict)
        :raises RestClientException: if response status code indicates
            request failure
        """
        status_code = response.status_code

        try:
            res_json = response.json()
        except json.decoder.JSONDecodeError:
            res_json = []

        if 200 <= status_code < 300:
            return status_code, res_json

        raise RestClientException("Request not successful, returned status code {}, response {}."
                                  .format(status_code, res_json))

    def fetch_and_lock(self, request_body=None):
        """
        """
        if request_body is None:
            if self._default_fetch_and_lock_rb is None:
                raise RestClientException("Request body is missing.")
            request_body = self._default_fetch_and_lock_rb

        fetch_and_lock_url = self._server_url + '/engine-rest/external-task/fetchAndLock'

        try:
            response = requests.post(fetch_and_lock_url, json=request_body)
        except Exception as err:
            raise RestClientException(err)
        else:
            status_code, json_data = self._process_response(response)
            self._last_res_json = json_data
            return status_code, json_data

    def get_last_fetch_and_lock_response(self):
        return self._last_res_json

    def complete(self, id_, request_body=None):
        """
        """
        if request_body is None:
            if self._default_complete_rb is None:
                raise RestClientException()
            request_body = self._default_complete_rb

        complete_url = self._server_url + '/engine-rest/external-task/{}/complete'.format(id_)

        try:
            response = requests.post(complete_url, json=request_body)
        except Exception as err:
            raise RestClientException(err)
        else:
            status_code, json_data = self._process_response(response)
            return status_code, json_data

    def get_process_definitions(self):
        """
        """
        try:
            response = requests.get(self._server_url + '/engine-rest/process-definition', json={})
        except Exception as err:
            raise RestClientException(err)
        else:
            status_code, json_data = self._process_response(response)
            return status_code, json_data

    def start_process_instance(self, definition_key, business_key, variables):
        """
        """
        request_body = {
            "variables": variables,
            "businessKey": business_key
        }

        start_process_instance_url = self._server_url + '/engine-rest/process-definition/key/' \
                                                        '{}/start'.format(definition_key)

        try:
            response = requests.post(start_process_instance_url, json=request_body)
        except Exception as err:
            raise RestClientException(err)
        else:
            status_code, json_data = self._process_response(response)
            return status_code, json_data


class RestClientException(Exception):
    pass
