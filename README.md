# COLTER

A decentralized, QoS-aware workload co-location framework based on lightweight continual
learning. Autonomous node-level agents predict interference and place tasks locally, with a
lightweight global coordination layer for cluster-level load balance.

This repository accompanies the manuscript *"COLTER: A Decentralized Framework for QoS-Aware
Workload Co-location with Lightweight Continual Learning"* (Journal of Parallel and Distributed
Computing, JPDC-D-25-01081).

## Scope and status of this repository

This is a **research-prototype subset** extracted from an internal production system built on
**Apache Hadoop YARN 3.0**, co-developed with an industry partner. Because of enterprise-compliance
constraints, parts of the end-to-end pipeline — some runtime metadata collection and the
production integration glue — cannot be fully open-sourced. Where the extraction left a call chain
incomplete, we have restored the **core decision logic** as reference implementations (marked in
comments) so the algorithms in the paper can be read and followed.

As a result, the code documents the design and core logic; it is **not a turnkey, fully buildable
deployment**. It depends on Hadoop/YARN 3.0 libraries that are not vendored here.

## Module layout

| Directory | Role (paper component) |
|---|---|
| `appcontroller/` | Node-level agent (ACo): metric collection, and layer-aware performance prediction (LAIP) |
| `analyzer/` | Resource-pattern detection and sensitivity classification (batch-type / layer analysis) |
| `continuouslearning/` | Temporal Evolution Forest (TEF): the lightweight continual-learning predictor |
| `recommendation/` | Recommendation scoring and confidence checking |
| `scheduler/` | QoS-aware co-scheduling, adaptive scheduling, threshold management, global coordination |
| `common/` | Shared types: `Metrics`, `WorkloadProfile`, `WorkloadTypes`, `Constants`, `Utils` |

## Baselines

Quasar and DRL-Sched were implemented and evaluated in the internal environment; their
configuration is described in the paper. The full baseline code is tied to the production
integration and is not part of this public subset.

## Experimental artifacts

`experiments/` contains the plotting scripts and the underlying measurement data behind the
main figures:

- `latency_analyse.py` + `tail_latency_data.csv` — per-service LRA tail latency (Fig. 5)
- `jct_analyse.py` + `spark_jct.csv` — batch job completion time (Fig. 7)

```
python3 experiments/latency_analyse.py
python3 experiments/jct_analyse.py
```

## Notes on specific implementation details

- **Metric normalization** (`analyzer/ResourcePatternDetector.java`): memory bandwidth and I/O
  throughput are scaled to an approximate `[0,100]` range relative to the platform's peak memory
  bandwidth (~115 GB/s per socket) and I/O baseline (~500 MB/s per node), so they are comparable
  with CPU/LLC percentages during batch-type classification.
- **Layer-aware prediction** (`appcontroller/PerformancePredictor.java`): the component type
  modulates the predicted interference via the dominant contended resource (highest
  `sensitivity × utilization`), using the per-layer sensitivities defined in
  `WorkloadTypes.createLraProfile`. The adjustment strength is a placeholder pending recalibration
  (see comments), as the original production constant is not available.
- **Application load** (`appcontroller/MetricsCollector.java`): measured as the node thread/task
  count from `/proc/loadavg`, a system-level signal that needs no application instrumentation.
