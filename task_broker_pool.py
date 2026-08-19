import argparse
from task_broker.worker import run as run_worker
from threading import Thread


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

    for i in range(args.worker_count):
        Thread(name=i, target=run_worker, args=(
            args.server_url,
            args.topic_name,
            args.polling_rate,
            args.host,
            args.port,
            i
        )).start()


if __name__ == "__main__":
    main()
