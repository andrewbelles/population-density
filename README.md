Population Density Prediction

Predicting population density across the contiguous United States using climate data, with plans for multi-modal fusion architectures.

## Current Status

Started with climate data regression models to establish baseline performance. The approach aggregates daily weather measurements (temperature, precipitation) by county and decade to predict population density for corresponding time periods.

### Data Processing Pipeline

The `aggregate_climpop.py` script processes:
- **Climate Data**: Daily GHCNd gridded data (temperature min/max/avg, precipitation)
- **Population Data**: Census data for 1960, 1990, 2020
- **Geography**: County boundaries and centroids for spatial modeling

Climate features are aggregated monthly by county, then split by decade to avoid temporal leakage. The pipeline outputs a hierarchical `.mat` file structure separating each decade's features/labels while caching shared coordinate data.

### Current Models

All benchmarks were performed on the `2020` climate dataset as a baseline. 

The baseline `models/linear_model.py` supports decade-specific training:
+ **Linear Regression**: Pure, linear baseline. For climate/population dataset: 
    - $r^2$: ~0.08, RMSE: ~0.79 

Likewise `models/forest_models.py` support decade-specific training:

+ **Random Forest**: Baseline ensemble model
  - $r^2$: ~0.26, RMSE: ~0.70
+ **XGBoost**: Gradient boosting with spatial coordinates as features
  - $r^2$: ~0.24, RMSE: ~0.71
+ **GPBoost**: Planned spatial Gaussian process + boosting hybrid

The Graph Neural Network uses the GeospatialModel class to support quick implementation of dataset specific GNNs. Relevant Source Files: 
+ `models/gnn_models.py`
+ `models/geospatial.py`
And the C++ modules defining the high-performance backend: 
+ `support/geography_graph.cpp/.hpp`

The model's performance on the `2020` baseline climate dataset.
+ **GNN**: Encode geospatial relationships through a geospatial adjacency graph with features.
   - $r^2$: ~0.33, RMSE: ~0.67

Faint positive variance for linear regression and an increase for more non-linear regressors supports evidence that more detailed/sophisticated datasets could provide substantial support to both multi-modal architectures. The GNN performance suggests that spatial encodings provide much needed support that the current climate dataset lacks.  

### Models Usage

```bash
# Set up environment
nix-shell

# Process raw data (creates climpop.mat)
python models/aggregate_climpop.py

# See baseline against linear model on a specific decade 
python models/linear_model.py --decade 2020 

# Train models on specific decades
python models/forest_models.py --decade 2020 --rf --xgb
``` 

## Supporting Modules 

### High-Performance C++ Geospatial Graph Backend 

The project includes a custom C++ GeospatialGraph class (located in `support/geography_graph.cpp`) optimized for spatial graph neural network operations. Some key features: 

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

```bash 
# Build project (in support/)
make 

# Run unit tests for geography_graph 
make test 

# Clean support modules 
make clean 

# Compile python bindings 
make python 

# Install python bindings into models/ 
make install-python 

# Clean just python bindings 
make clean-python 
```

And the backend can be imported via `import geography_graph_cpp`. 

## Future Plans

Model Architecture Evolution

Early Fusion: Train category-specific encoders (climate, satellite, socioeconomic) into shared latent space for unified regression.

Late Fusion: Train specialized models per data category, then learn weighted combinations or meta-models over individual predictions.

### Additional Data Categories

- Nighttime Satellite Imagery: CNN-based features from satellite lighting data
- Socioeconomics: Land costs, zoning, income, employment statistics
- Graph Networks: County adjacency graphs with centroid-based spatial relationships

### Technical Improvements

Climate data will expand beyond basic temperature/precipitation to include humidity, weather extremes, and seasonal patterns. Spatial modeling will incorporate county adjacency graphs for GraphNN approaches.

The current climate-only models perform poorly as expected, primarily because you wouldn't (and shouldn't) predict population density from just weather data. But these features should prove valuable in the larger multi-modal architectures.
