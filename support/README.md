# Support

This directory contains:
- The C++ geospatial backend (`GeospatialGraph`) used for graph construction and distance computation.
- Python helper utilities used across models and analysis (`support/helpers.py`).

## C++ Backend

Primary sources:
- `support/geospatial_graph.cpp`
- `support/geospatial_graph.hpp`
- `support/python_bindings.cpp`

### Build

From the project root:
```bash
make -C support
make -C support test
make -C support python
make -C support install-python
```

The Python extension can then be imported as:
```python
import geospatial_graph_cpp
```

## Python Helpers

`support/helpers.py` contains:
- `project_path(...)` (uses `PROJECT_ROOT` when set; `shell.nix` exports it)
- dataset loaders returning the invariant `DatasetDict = {"features","labels","coords"}`
- scaling helpers (`fit_scaler`, `transform_with_scalers`)
- splitting helpers (`split_indices`, `kfold_indices`)

