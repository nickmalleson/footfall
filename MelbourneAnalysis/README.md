# Predicting Pedestrian Footfall: <br /> Location Data Analysis of Melbourne, Australia

This repository contains code for building a machine learning model to predict footfall across the city of Melbourne.  
The aim is to be able to predict footfall at any location in the city, for a particular time or day.  
Sensors across the city record hourly counts of pedestrians, and we use this data combined with other descriptors of the built environment likely to drive footfall patterns.  

## Table of Contents
* [Downloading data](#downloading-data)
* [1. Preparing data](#preparing-data)
* [2. Analysing data](#analysing-data)
* [3. Modelling](#modelling)

<a name="downloading-data"></a>
## Downloading data

Footfall data is downloaded by selecting 'Export --> CSV' from: 
* https://data.melbourne.vic.gov.au/Transport/Pedestrian-Counting-System-Monthly-counts-per-hour/b2ak-trbp 

Weather data is downloaded from: https://www.timeanddate.com/weather/australia/melbourne/historic. 

Data on other features is downloaded from: https://data.melbourne.vic.gov.au/. Including data on:
* [Lighting](https://data.melbourne.vic.gov.au/City-Council/Feature-Lighting-including-light-type-wattage-and-/4j42-79hg)
* [Landmarks](https://data.melbourne.vic.gov.au/People/Landmarks-and-places-of-interest-including-schools/j5vt-ppat)
* [Street furniture](https://data.melbourne.vic.gov.au/widgets/8fgn-5q6t?mobile_redirect=true)
* [Buildings](https://data.melbourne.vic.gov.au/Property/Buildings-with-name-age-size-accessibility-and-bic/pmhb-s6pn)
* [Bike share dock locations](https://data.melbourne.vic.gov.au/w/vrwc-rwgm/spy9-nmud?cur=l0YdZo6QE_m&from=88D7wUgzKYw)

Future work coulc consider other possible interesting sources of data:
* Social indicators (affluence of area?)
* Tree canopies (green space?
* Live music venues
* Playgrounds
* Cafes/restaurants/bistro seats/Bar, tavern, pub patron capacity
* Public toilets
* Population of area, broken down further demographically. Using: australian bureau of statistics (abs) census of population and housing. A
* Number of jobs, collected through the City of Melbourne’s Census of Land Use and Employment (CLUE) 

<a name="preparing-data"></a>
## 1. Preparing data
### <ins> 1. Clean data </ins> 
Scripts for cleaning the data for: footfall; other features; transport features; and special data variables.
This includes getting the data all into the same format.  
For some features (e.g. buildings) it also involves consolidating sub-type categories.  
Cleaned versions are saved to the Cleaned_data directory:

The outputs from this are:
* bikes_clean.csv - columns: station_id, capacity, latitude, longitude
* buildings_clean.csv - columns: year, n_floors, building_type, access_type, access_rating, latitude, longitude
* landmarks_clean.csv - columns: theme, subtheme, featurename, latitude, longitude
* lights_clean.csv - columns : lamptype_lupvalue, lamp-rating_w, latitude, longitude
* street_inf_clean.csv - feature, condition_rating, latitude, longitude
* weather_data_{year}.csv - datetime (hourly), Temp,	Humidity,	Pressure,	Rain, (binary 1 or 0),	WindSpeed

### <ins>  2. LinkSpatialFeaturesToSensors </ins> 
This finds the number of features of each type, and the number of features of each subtype, in a buffer of Xm of each sensor. 
This is done for a number of different buffer sizes to allow testing of the buffer size which leads to the best results. This produces:

* num_features_near_sensors_{buffer_size_m}.csv - each column contains data for one sensor, rows specify the feeature types 
* feature_subtypes_near_sensors_{buffer_size_m}.csv - each column contains data for one sensor, rows specify the feeature subtypes

### <ins>  3. ProcessStreetNetworkData </ins>
This is for calculating the betweenness of the street network.

### <ins> 4. ProcessWeatherData </ins>
Script which scrapes the weather data from the Melbourne historic weather website, cleans the data, and saves yearly weather csvs to the Cleaned_data directory:
* ScrapingWeatherData.ipynb

### <ins> 5. PrepareDataForModelling </ins> 
Join cleaned datasets together to get footfall data alongside the predictor variables.  
Add dummy variables for day of week and month AND a sin/cos representation of each as cyclical

<a name="analysing-data"></a>
## 2. Analysing data

Looking at distributions of predictor and predictand variables. Checking whether the missing footfall values are

<a name="modelling"></a>
## 3. Modelling
###  <ins> 1. ModelSelection.ipynb  </ins>
Tests the performance of a number of different machine learning models using 10-fold cross validation. These include:
* Linear regression
* Random Forest
* XGBoost
* Extra Trees Regressor

The outputs of the 10-fold cross validation process are:
* The error metric scores associated with that model (averaged over all folds)
    * The MAE, the MAPE and the RMSE
### <b> Conclusion: Random Forest Regressor is best performing model </b>

There was another version of this script where Year was not included as a variable (this is now deleted as decided it shouldn't be included)

###  <ins> 2. ModelSelection_TestBufferSize.ipynb </ins>
Tests the performance of a random forest regressor using features collected within a number of different buffer sizes: 50,100,200,300,400,500,600,1000

The outputs of the 10-fold cross validation process are:
* The error metric scores associated with that model (averaged over all folds)
    * The MAE, the MAPE and the RMSE
### <b> Conclusion: 500m buffer results in best performing model </b>

###  <ins> 3. ModelEvaluation.ipynb  </ins>
Tests the performance of a random forest regressor using features collected within 500m.

This is validated using a simple 80 - 20 train test split with the chronological order of the data preserved.  
The outputs of the validation process are:
* The error metric scores associated with that model (averaged over all folds)
    * The MAE, the MAPE and the RMSE
* The Gini feature importance associated with each fold
* The model used in each fold, from which the permutation imporance associated with each fold can be calculated
* A prediction for each data point in the dataset (within cross validation each data point is included in the test set only once and thus despite their beng multiple cross-validation folds, each true value of Y has only one associated prediction )

Also breaks down prediction and prediction error by different time slices and different parts of city. 

###  <ins> 4. FittingFinalModel.ipynb </ins>
Fit a Random Forest Regressor with a 500m buffer on the whole dataset. Saves this model to a pickle file so it can be reused in future. Also saves the predictor variables used in constructing it, and the real values associated with those predictor variables.

###  <ins> 5. AssessingFinalmodel.ipynb  </ins>
Reads in the Random Forest model fitted on the whole dataset from the pickle file. 
Find the Gini and Permutation feature importances returned from this final fitted model.  
There are no predicted values to plot, because we fitted the model on the whole dataset.

