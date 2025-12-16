# Population Density Prediction

Predicting population density across the contiguous United States using climate data, with plans for multi-modal fusion architectures.

## Current Status

Started with climate data regression models to establish baseline performance. The approach aggregates daily weather measurements (temperature, precipitation) by county and decade to predict population density for corresponding time periods.

### Recent Work (Post-README)

- Implemented a repeated cross-validation interface (`analysis/cross_validation.py`) with fold-safe scaling to avoid transductive leakage.
- Added multi-output regression support (notably used for coordinate→climate prediction and residual workflows).
- Added a residual export + reload workflow to build downstream datasets from first-stage residuals (`analysis/residual_analysis.py`).
- Added a county-level “sophisticated” climate view based on NOAA NClimDiv products (Palmer indices + degree days) and compilation scripts.

### Data Processing Pipeline

This repo currently supports two climate pipelines:

1) Legacy gridded daily pipeline (older baseline)
- **Climate Data**: Daily GHCNd gridded data (temperature min/max/avg, precipitation)
- **Population Data**: Census data for 1960, 1990, 2020
- **Geography**: County boundaries and centroids for spatial modeling

2) NOAA NClimDiv county pipeline (current “sophisticated view”)
- **Climate Data**: NClimDiv county products (Palmer indices + degree days; per-month, per-county)
- **Population/Geography**: Census + Gazetteer, aligned by county FIPS
- Outputs are compiled by `preprocessing/` scripts into `.mat` files under `data/`

Climate features are aggregated monthly by county, then split by decade to avoid temporal leakage. The pipeline outputs a hierarchical `.mat` file structure separating each decade's features/labels while caching shared coordinate data.

### Current Models

All benchmarks below should be treated as placeholders while the new benchmark harness is being finalized.

The baseline `models/linear_model.py` supports decade-specific training via cross-validation:
+ **Linear Regression**: Pure, linear baseline. For climate/population dataset: 
    - $r^2$: $-0.482 \in [-0.749, -0.214]$, RMSE: $2491.7 \in [2328.2, 2655.2]$ 

Likewise tree models are exposed as standalone modules:

+ **Random Forest** (`models/random_forest_model.py`): 
    - $r^2$: $-0.051 \in [-0.263, +0.162]$, RMSE: $2070.02 \in [1914.4, 2225.6]$ 
+ **XGBoost** (`models/xgboost_model.py`):
    - $r^2$: $-0.050 \in [-0.186, +0.086]$, RMSE: $2070.02 \in [2081.4, 2407.9]$ 
+ **GPBoost** (`models/gp_xgboost_model.py`). I have not ran benchmarks I'm particularly confident in. But from what I've seen it doesn't perform great (but that could also be a me problem) 

The Graph Neural Network uses the GeospatialModel class to support quick implementation of dataset specific GNNs. Relevant Source Files: 
+ `models/gnn_models.py`
+ `models/geospatial.py`

And the C++ modules defining the high-performance backend: 
+ `support/geospatial_graph.cpp/.hpp`

The model's performance on the `2020` baseline climate dataset has not been computed.
+ **GNN**: Encode geospatial relationships through a geospatial adjacency graph with features. I haven't written the GNN to work with the `CrossValidator` so as I just said, I don't have a confident statement about its performance. 

All climate models are very poor predictors of population density as the only features. This gave much needed direction. Either a climate-aware coordinate encoding could be made, or the climate features altogether could be forgone (depending on whether climate provides anything the coordinates do not). 

### Models Usage

Build, data download, dataset compilation, and analysis harness usage are documented in:
- `scripts/README.md`
- `preprocessing/README.md`
- `analysis/README.md`
- `models/README.md`

Quick start (NClimDiv county pipeline):
```bash
nix-shell
./scripts/fetch_nclimdiv_county.sh
./scripts/parse_nclimdiv_county.sh 1990 2020
python preprocessing/climate_population_dataset.py
python analysis/cross_validation.py --decade 2020 --folds 5 --repeats 20 --models xgb
```

## Supporting Modules 

### High-Performance C++ Geospatial Graph Backend 

The project includes a custom C++ GeospatialGraph class (located in `support/geospatial_graph.cpp`) optimized for spatial graph neural network operations. Some key features: 

Distance computation: Graph uses the Haversine distance (or distance between two points on $S^2$ given their latitude and longitude) as its standard metric to create an accurate/stable distance matrix for all counties centroids. 

Flexible adjacency metrics: 
- K-Nearest Neighbors (KNN): An edge has at most its $k$ nearest neighbors where nearest is defined by the standard metric. 
- Distance-Bounded: Connect counties within a specific distance threshold (km) 
- Complete Graph: Connect counties without restriction, sorted by distance. 

PyTorch Integration: Python bindings (`python_bindings.cpp`) provides direct access for GNN frameworks:
- Edge indices as source-target pairs for `torch_geometric`.  
- Distance vectors for edge weights. 
- County coordinate extraction for positiional encodings. 

The C++ implementation can process 3,200+ US counties efficiently in a highly configurable manner. 

### Support Usage 

Support build instructions are maintained in `support/README.md`.

And the backend can be imported via `import geospatial_graph_cpp`. 

## Future Plans

Model Architecture Evolution

Early Fusion: Train category-specific encoders (climate, satellite, socioeconomic) into shared latent space for unified regression.

Late Fusion: Train specialized models per data category, then learn weighted combinations or meta-models over individual predictions.

### Additional Data Categories

- Nighttime Satellite Imagery: CNN-based features from satellite lighting data
- Socioeconomics: Land costs, zoning, income, employment statistics

### Technical Improvements

Climate data will expand beyond basic temperature/precipitation to include humidity, weather extremes, and seasonal patterns. 

The current climate-only models perform poorly as expected, primarily because you wouldn't (and shouldn't) predict population density from just weather data. But these features should prove valuable in the larger multi-modal architectures.
