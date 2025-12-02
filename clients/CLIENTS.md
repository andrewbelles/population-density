# Api Clients 

## Climate Data  

Data I am collecting using this agent: 
- Temperature 
    - Average, Maximum, Minimum 
- Precipitation 
    - Rainfall, Snowfall, Annual and Seasonal
- Drought Indices 
    - Palmer Drought Severity Index 
- Heating and Cooling Degree Days 
- Extreme Weather/Climate 
    - Frequenc of heat waves, cold waves, heavy rainfall, etc. 
- Sunshine 
- Humidity

Datasets from NOAA that I specifically hit to gather this data: 
- GHCN-D, a comprehensive collection of daily weather observations from land stations 
    - TMAX, TMIN, PRCP, SNOW, SNWD (Snow depth)
- nClimGrid 
    - Interpolated climate data on a regular grid covering at ~5km resolution 
    - Includes daily max/min/avg temperature and precipitation. 
    - Contiguous US 
    - Monthly: Spans from Jan 1895-Present and is updated monthly 
    - Daily: Spans Jan 1, 1951-Present and is updated daily 
    - Derived from GHCN-D dataset already. Interpolates beyond a naive interpolation using domain knowledge about geospatial information, etc. 
- Climate Normals (NOAA): 
    - 30 year averages from 1991-2020, computed for each station and each day, month, season, and year. 
    - Normals cover all standard climatogical variables
        - temperature, precipitation, snowfall, snow depth, humidity, wind, cloudiness 
    - Includes extremes and frequencies (beyond just averages)
    - Gridded Normals are available, similar to nClimGrid which is great 
- nCLIMDIV (NOAA).
    - State/County level resolution 
    - Precomputed monthly values for each climate division 
    - This is the data set containing drought indices, and heating/cooling days

Issues: 
- nClimGrid is a not a JSON api that can be requested at. Fortunately it looks like NCEI exposes an API that lets you request a latitude, longitude bounding box for information. 
