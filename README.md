# Risk Engine

gRPC-based risk calculation service that dynamically loads models from the `models/` directory, sits behind Envoy proxies for HTTP/2 routing and load balancing, and optionally integrates with Camunda BPM for workflow-driven execution. Observability is provided via StatsD → Prometheus → Grafana.

## Architecture

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                     Docker Compose                       │
                    │                                                         │
  Client (gRPC) ───►│  front-envoy (:10000)                                   │
                    │       │  round-robin / STRICT_DNS                       │
                    │       ▼                                                 │
                    │  service1 × N                                           │
                    │  ┌─────────────────────────────┐                        │
                    │  │ service-envoy (:80)         │                        │
                    │  │      ▼                      │                        │
                    │  │ RiskEngine gRPC (:8080)     │◄── models volume       │
                    │  │  └─ models.<name>.model     │                        │
                    │  └─────────────────────────────┘                        │
                    │                                                         │
                    │  models (init) ── copies /models into shared volume     │
                    │                                                         │
                    │  Envoy stats ──► statsd_exporter (:9125/UDP, :9102)     │
                    │                       │                                 │
                    │                       ▼                                 │
                    │                 prometheus (:9090)                      │
                    │                       │                                 │
                    │                       ▼                                 │
                    │                   grafana (:3000)                       │
                    └─────────────────────────────────────────────────────────┘

  Optional BPM path (outside compose by default):

  Camunda BPM ──fetch/lock──► task_broker workers ──gRPC──► front-envoy
       ▲                            │
       └──────── complete ──────────┘
```

### Components

| Component | Role |
|-----------|------|
| **front-envoy** | Edge proxy; listens on `:10000` (gRPC/HTTP2), admin on `:8001`. Routes all traffic to the `service1` cluster with round-robin DNS load balancing. Emits StatsD metrics with prefix `front-envoy`. |
| **service1** | Risk Engine node: Envoy sidecar (`:80` → local gRPC `:8080`) + Python `RiskEngine` servicer. Scales horizontally via `docker-compose scale`. Admin port `:8081`. Emits StatsD metrics with prefix `service1`. |
| **models** | Init-style container that populates a shared Docker volume with model code/data for service instances. |
| **statsd_exporter** | Receives Envoy StatsD metrics; exposes them in Prometheus format on `:9102`. |
| **prometheus** | Scrapes `statsd_exporter` every 5s. |
| **grafana** | Pre-provisioned Prometheus datasource and Envoy dashboard. |

### gRPC API

Defined in `service/risk.proto`:

```protobuf
service RiskEngine {
    rpc Calculate (CalculationRequest) returns (CalculationResponse) {}
}

message CalculationRequest {
    string model_name = 1;
    bytes model_parameters = 2;  // model-specific protobuf
    bytes model_input = 3;       // model-specific protobuf
}

message CalculationResponse {
    map<string, double> values = 1;
}
```

1. Client sets `model_name` to a folder under `models/` (e.g. `sum`, `logistic_regression`).
2. `model_parameters` / `model_input` are serialized model-specific messages from that model's `api.proto`.
3. The service lazily imports `models.<name>.model` and calls `calculate(parameters, input)`.
4. Optional `decode_parameters` / `decode_input` on the model module deserialize the byte payloads.

### Model loading

`ModelLoader` in `service/service.py` validates names (`^[a-z_][a-z0-9_]*$`), imports modules on first use, and caches them. Unknown or invalid names raise errors returned to the client.

### BPM / Camunda integration

Workers under `task_broker/` poll Camunda external tasks, convert process variables into model protobufs (via per-model `converter.py` modules), call the Risk Engine over gRPC, and complete the task with the result.

Models that ship a `converter.py` (usable from BPM workers): `logistic_regression`, `scoring_card`, `scoring_to_pd_src`, `src_to_price`, `risk_based_pricing`.

Related scripts:

- `task_broker/worker.py` — single worker
- `task_broker_pool.py` / `start_worker_pool.sh` — multi-threaded worker pool
- `trigger_model.py` — start Camunda process instances with JSON variables
- `logistic_regression_workers/` — specialized LR workers and sample BPMN diagrams
- `task_broker/bpm_diagrams/` — sample BPMN process definitions

---

## Build and deployment

### Prerequisites

- Docker and Docker Compose
- (Optional, for local clients/codegen) Python 3 with packages from `service/requirements.txt`

### Start the stack

```bash
docker-compose up --build
# or detached:
docker-compose up --build -d
```

Stop:

```bash
docker-compose down
```

List containers / IPs:

```bash
docker-compose ps
./docker_ip.sh
```

### Images

| Dockerfile | Image purpose |
|------------|---------------|
| `Dockerfile-frontenvoy` | Envoy Alpine + `front-envoy.yaml` |
| `Dockerfile-service` | Envoy + Python Risk Engine, clients, codegen at build time |
| `Dockerfile-models` | Alpine copy of `models/` into the shared volume |

Service container startup (`service/start_service.sh`): starts `service.py` on port `8080`, then Envoy with `service-envoy.yaml`. Environment variables:

- `SERVICE_NAME` — used in Envoy `--service-cluster` / `--service-node` (default `1` in compose)
- `MAX_WORKERS` — documented for the gRPC thread pool (compose sets `10`)

### Scaling

```bash
docker-compose scale service1=2
```

Front Envoy resolves `service1` via `STRICT_DNS` and load-balances round-robin across all instances.

### Local codegen / iris train (optional)

```bash
make          # trains iris model if needed, runs protobuf codegen
make clean    # removes generated *_pb2*.py, caches, iris pickle
```

Protobuf stubs are also generated inside the service image via `service/codegen.py`.

### Calling the service

Through the front proxy (recommended):

```bash
python3 ./client_sync.py --host <front-envoy-ip> --port 10000
```

Directly to a service sidecar:

```bash
python3 ./client_sync.py --host <service1-ip> --port 80
```

From inside a service container:

```bash
docker exec <service1-container> python3 /app/client_sync.py --port 8080 --host localhost
```

Other clients:

- `client_async.py` — async gRPC futures demos (`sum`, `div`, `avg`, `iris`, `option_price`)
- `client_async_with_lr.py` — async client including logistic regression

### Exposed ports

| Port | Service | Purpose |
|------|---------|---------|
| `10000` | front-envoy | gRPC ingress |
| `8001` | front-envoy | Envoy admin |
| `80` | service1 (internal) | Service Envoy → gRPC |
| `8081` | service1 | Service Envoy admin |
| `9125/udp`, `9102` | statsd_exporter | StatsD in / Prometheus metrics out |
| `9090` | prometheus | Prometheus UI / API |
| `3000` | grafana | Grafana UI |

---

## Available models (`models/`)

Each loadable model is a package with `model.py` (implements `calculate`) and typically `api.proto` (request schemas). Name used in `CalculationRequest.model_name` equals the directory name.

### Demo / utility

| Model | Description |
|-------|-------------|
| **sum** | Sums a list of doubles. Returns `result`. |
| **div** | Divides `dividend` by `divisor`. Returns `result`, `quotient`, `remainder`. |
| **avg** | Arithmetic or geometric mean of a value list (selected via `ModelParams.type`). Returns `total`, `count`, `result`. |
| **iris** | Sklearn classifier predicting Iris species from sepal/petal measurements. Trained via `models/iris/train.py`; artifact `iris_model.pickle`. Returns `species`. |

### Credit risk & pricing

| Model | Description |
|-------|-------------|
| **logistic_regression** | Probability of default (PD) via WOE-binned risk attributes and a pre-trained logistic regression (`data/lr_model.json`, `data/woe_bins.json`). Input: key/value risk attributes. Returns `result` (PD). |
| **scoring_card** | Credit score from categorical attributes using a YAML scoring card (`parameters.scoring_card_name` → file under `data/`). Supports single and combined (tuple) variable keys. Returns `result` (score). |
| **scoring_to_pd_src** | Maps a credit score to PD and SRC using a CSV reference scale (`parameters.reference_scale_name`). Returns `result` (SRC) and `probability_of_default`. |
| **src_to_price** | Maps SRC to a product price via a conversion table CSV (`parameters.conversion_table_name`). Returns `result` (price). |
| **risk_based_pricing** | From a PD value, computes cost baseline (funds + operations + cost of risk using LGD / rates from `data/pricing_model.csv`) and a risk rating (`data/pd_to_risk_rating.csv`). Returns `result` (cost) and `risk_rating_int`. |

Typical credit pipeline (also reflected in BPM diagrams):

```
risk attributes → logistic_regression (PD)
                 → scoring_card → scoring_to_pd_src → src_to_price
                 → risk_based_pricing (cost / rating from PD)
```

### Markets / portfolio

| Model | Description |
|-------|-------------|
| **option_price** | European vanilla option price with continuous dividend (Black–Scholes). Inputs: spot `S`, strike `K`, maturity `T`, rate `r`, dividend yield `q`, volatility `sigma`, `option` (`call`\|`put`). Returns `price`. |
| **portfolio_var** | Monte Carlo Value-at-Risk for a multi-asset portfolio (correlated returns from historical prices, Cholesky / multivariate normal). Parameters: `number_of_assets`, `alpha` (percentile), `time_perspective`. Input: flat `asset_prices` and `asset_weights`. Returns `result` (VaR). |
| **portfolio_optimization** | Optimizes portfolio weights: minimize variance or maximize expected return / SD (minimize negated ratio). Algorithms: `TRUST_CONSTR`, `SLSQP`, `COBYLA`. Supports per-weight bound constraints. Uses SciPy `optimize.minimize`. |

### Not wired as a Risk Engine model

| Folder | Notes |
|--------|-------|
| **digits** | Standalone handwritten digit recognition sample (OpenCV / sklearn). Has its own README; no `model.py` / gRPC `api.proto` for the Risk Engine. |

Shared helper: `models/utils.py` (`ProtoDecoder` for request bytes).

---

## Monitoring (Prometheus & Grafana)

### Metrics pipeline

```
Envoy (front + service)  --StatsD/TCP-->  statsd_exporter:9125
                                              │
                                              ▼ scrape :9102
                                         prometheus:9090
                                              │
                                              ▼ datasource
                                          grafana:3000
```

Envoy configs (`front-envoy.yaml`, `service/service-envoy.yaml`) define a `statsd-exporter` cluster and `envoy.statsd` sinks with prefixes `front-envoy` and `service1`.

### Prometheus (`prometheus/config.yaml`)

- Global scrape interval: **15s**
- Job `statsd`: scrapes `statsd_exporter:9102` every **5s**, label `group: services`
- Config mounted at `/etc/prometheus.yaml`; process started with `--config.file=/etc/prometheus.yaml`

UI: `http://<prometheus-host>:9090`

### Grafana

Provisioned from compose mounts:

| File | Purpose |
|------|---------|
| `grafana/grafana.ini` | Instance name; admin user/password **`admin` / `admin`** |
| `grafana/datasource.yaml` | Prometheus datasource at `http://prometheus:9090` (default) |
| `grafana/dashboard.yaml` | File provider loading dashboards from `/etc/grafana/provisioning/dashboards/` |
| `grafana/dashboard.json` | Prebuilt Envoy metrics dashboard (2xx/4xx/5xx rates, latency, etc., templated by source/destination clusters) |

UI: `http://<grafana-host>:3000` — log in with `admin` / `admin`.

### Envoy admin

Useful for live cluster/stats inspection:

- Front: `http://<front-envoy-host>:8001`
- Service: `http://<service1-host>:8081` (per replica)

---

## Project layout

```
├── docker-compose.yml          # Full stack definition
├── front-envoy.yaml            # Edge Envoy config
├── Dockerfile-*                # frontenvoy / service / models images
├── service/                    # Risk Engine + service Envoy
│   ├── service.py              # gRPC servicer + model loader
│   ├── risk.proto              # Public API
│   ├── service-envoy.yaml
│   └── start_service.sh
├── models/                     # Pluggable calculation models
├── task_broker/                # Camunda ↔ gRPC workers
├── prometheus/                 # Scrape config
├── grafana/                    # Datasource, dashboard provisioning
├── client_*.py                 # Example gRPC clients
├── trigger_model.py            # Start Camunda processes
└── tests/models/               # Unit tests for selected models
```

## Adding a new model

1. Create `models/<name>/` with `api.proto` and `model.py`.
2. Implement `calculate(parameters, data)` returning a `dict` of string → float (for `CalculationResponse.values`).
3. Optionally define `decode_parameters` / `decode_input` using `models.utils.ProtoDecoder`.
4. For BPM: add `converter.py` with `convert_to_model_parameters` / `convert_to_model_input`.
5. Regenerate stubs (`make` or rebuild the service image) and place any data files under the model directory (they are copied via the `models` volume).

## Tests

```bash
# example: model unit tests
python -m pytest tests/models/
```

Covered under `tests/models/`: `logistic_regression`, `portfolio_optimization`, `portfolio_var`, `risk_based_pricing`.

## Notes

- The service currently applies a random simulated latency (0–5s) per `Calculate` call for demo purposes (`service/service.py`).
- Protocol Buffers Python implementation is forced to `'python'` in startup scripts for compatibility with the pinned `grpcio` stack.
- Default Grafana credentials are for local/dev use only; change them before any shared or production deployment.
