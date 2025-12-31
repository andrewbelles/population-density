# Testbench

End‑to‑end evaluation scripts for stacking, graph metrics, and downstream tests.

## Stacking + Correct‑and‑Smooth

`testbench/stacking.py` runs:
- Expert optimization per dataset.
- Stacked classifier on OOF probabilities.
- Correct‑and‑Smooth post‑processing on graph adjacency.

Example:
```bash
python testbench/stacking.py --resume
```

## Config caching:

- `testbench/model_config.yaml`

## Graph Metrics

`testbench/graph_metrics.py` evaluates adjacency diagnostics and metric learners.

## Downstream Metric

`testbench/downstream_metric.py` loads a trained metric (from YAML) and optimizes Correct‑and‑Smooth on
top of it. Furthermore, optimizes the learned metric space with the Correct-And-Smooth model's accuracy as the downstream target. 
