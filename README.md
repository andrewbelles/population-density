# Multiview Learning of Data Categories as Manifolds 

## Data Categories 

I will pull data from 8-10 various parent categories that offer underlying information about change in population density. 

Key data (abstractly stated): 
- Demographic data 
    - Census data 
- Building footprint/density 
- Nighttime light intensity 
- Land Use/Cover 
- Transportation Networks
- Socioeconomic
    - Housing density 
    - Economic activity 
- Cell phone Towers? 

## C++ API Crawler 

I intend to implement a highly flexible API Crawler Class which aims to make API calls for specific to the instance. In a most basic sense, the Crawler will implement methods for hitting the specific API endpoint, support for retrying, exception handling for errors, and support for calling user defined functions for handling parsed JSON. The Crawler will be designed to work for RESTFUL, Json API. 

Json data will be pulled and a handler will insert the parsed data into a SQLite database. The end goal will be for each category to have a single (or possibly multiple) Crawler(s) that utilizes a CPU Scheduler-esque algorithm to alternate calls between Crawlers. 

## Crawler Logging 

A detailed logger will be implemented that aims to log Crawler failures or  
