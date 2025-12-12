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

The baseline `models/linear_model.py` supports decade-specific training:
+ **Linear Regression**: Pure, linear baseline. For climate/population dataset: 
    - $r^2$: ~0.08, RMSE: ~0.79 

Likewise `models/forest_models.py` support decade-specific training:

+ **Random Forest**: Baseline ensemble model
  - $r^2$: ~0.26, RMSE: ~0.70
+ **XGBoost**: Gradient boosting with spatial coordinates as features
  - $r^2$: ~0.24, RMSE: ~0.71
+ **GPBoost**: Planned spatial Gaussian process + boosting hybrid

Faint positive variance for linear regression and an increase for more non-linear regressors supports evidence that more detailed/sophisticated datasets could provide substantial support to both multi-modal architectures. 

### Usage

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
