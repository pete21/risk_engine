#!/bin/sh
PORT=8080
#python3 /app/service.py --max_workers ${MAX_WORKERS} --port ${PORT} &
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python' python3 /app/service.py --port 8080 &
envoy -c /etc/service-envoy.yaml --service-cluster service${SERVICE_NAME} --service-node service${SERVICE_NAME}
