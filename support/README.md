# Support

C++ geospatial backend and shared helpers.

## C++ Backend

Primary sources:
- `support/geospatial_graph.cpp`
- `support/geospatial_graph.hpp`
- `support/python_bindings.cpp`

### Build

```bash
make -C support
make -C support test
make -C support python
make -C support install-python
```

Import:

import `graph_cpp`

## Python Helpers

`support/helpers.py` provides:

- Path helpers (project_path)
- Dataset loading conventions
- Scaling / splitting utilities
