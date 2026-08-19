import argparse
from collections import deque
from threading import Event, Thread
from time import sleep

from task_broker.light_worker import run as run_worker
from task_broker.bpm_rest_client import RestClient, RestClientException
from task_broker.grpc_client import GrpcClient


def run(bpm_server_url, topic_name, polling_rate, grpc_host, grpc_port, worker_count):

    bpm_client_dict = {}
    recv_event_dict = {}
    idle_workers_queue = deque()

    risk_engine_client = GrpcClient(grpc_host, grpc_port)

    for worker_id in range(worker_count):

        bpm_client_dict[worker_id] = RestClient(bpm_server_url)
        bpm_client_dict[worker_id].prepare_default_request_bodies(worker_id, topic_name)
        recv_event_dict[worker_id] = Event()

        Thread(name=str(worker_id), target=run_worker, args=(
            bpm_client_dict[worker_id],
            risk_engine_client,
            recv_event_dict[worker_id],
            worker_id,
            idle_workers_queue
        )).start()

    delay = 1.0 / abs(polling_rate)

    while True:

        try:
            worker_id = deque.pop(idle_workers_queue)
        except IndexError:

            print("All workers busy, waiting {} seconds.".format(delay))

            sleep(delay)
            continue

        while True:
            try:
                status_code, res_json = bpm_client_dict[worker_id].fetch_and_lock()
            except RestClientException as err:

                print("Error calling fetch and lock ({}).".format(err))

                sleep(delay)
                continue

            print("Fetch and lock status code is {}, response is {}.".format(status_code, res_json))

            if len(res_json) != 0:
                recv_event_dict[worker_id].set()
                sleep(delay)
                break

            sleep(delay)


def main():
    parser = argparse.ArgumentParser(description="Start worker pool.")
    parser.add_argument(
        "topic_name",
        type=str,
        help="topic name"
    )
    parser.add_argument(
        "--worker-count",
        type=int,
        help="worker count (default 2)",
        default=2
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

    run(args.server_url, args.topic_name, args.polling_rate, args.host, args.port,
        args.worker_count)


if __name__ == "__main__":
    main()
