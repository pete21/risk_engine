Usage:


docker-compose up [--build]
-d[etached] is optional, without it, you will see the container consoles output

docker-compose down


List of containers:
docker-compose ps


Container IP addresses:
./docker_ips.sh


Risk-engine request sent through front proxy:
python3 ./client_sync.py --port 10000 --host IP_risk-engine_front-envoy_1

directly through to the risk-engine service proxy:
python3 ./client_sync.py --port 80 --host IP_risk-engine_service1_1

other options (from inside the container):
docker exec risk-engine_service1_1 python3 /app/client_sync.py --port 8080 --host localhost 


Scaling up of the service1 (risk-engine) cluster:
docker-compose scale service1=2

After service1 cluster is scaled up to multiple service nodes, round robin kicks in using the dns (load balancing across all service1 nodes).

Monitoring:
Porty 8081 - service1 containers
Port 8001 - front-envoy containers


Prometheus/Grafana:

IP_Prometheus:9090
IP_Grafana:3000


