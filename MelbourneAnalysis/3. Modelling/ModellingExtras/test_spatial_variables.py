#!/usr/bin/env python
# coding: utf-8

# ## if include sensor_id then can fit quite a good model - but this would be a model that would only be able to predict at those locations

# In[15]:


import pandas as pd
# from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
# import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor 
# import xgboost as xgb
from sklearn.pipeline import Pipeline
# import folium
# import branca.colormap as cm
# from eli5.sklearn import PermutationImportance
# import joblib
# import os
# import psutil
# import geopy.distance
import sys

sys.path.append('../')
from Functions import *


# In[16]:


buffer_size_m = 400
input_csv =f"../../Cleaned_data/FormattedDataForModelling/formatted_data_for_modelling_allsensors_{buffer_size_m}_outlierremovaleachsensor.csv"


# ## Run models with cross-validation

# ### Define the error metrics for the cross-validation to return, and the parameters of the cross validation

# In[17]:


error_metrics = ['neg_mean_absolute_error', 'r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_percentage_error']
cv_parameters = KFold(n_splits=10, random_state=1, shuffle=True)


# In[18]:


rf_model_pipeline = Pipeline(steps=[['scaler',StandardScaler()],['rf_regressor', RandomForestRegressor(random_state = 1, n_jobs = 10)]])


# ### Prepare data for modelling

# In[19]:


Xfull, Yfull, data_time_columns = prepare_x_y_data(input_csv)


# ### Cut off data post-Covid

# In[20]:


Xfull= Xfull[0:2643750]
Yfull= Yfull[0:2643750]
data_time_columns = data_time_columns[0:2643750] # end of 2019


# ### Add sensor ID

# In[21]:


sensor_ids = pd.read_csv(input_csv)['sensor_id']
sensor_ids= sensor_ids[0:2643750]
Xfull['sensor_id'] = sensor_ids


# ### Choose which month_num and weekday_num option to include

# In[22]:


# If using the dummy variables
# Xfull.drop(['Cos_month_num', 'Sin_month_num', 'Cos_weekday_num', 'Sin_weekday_num'], axis=1)
# If using the cyclical variables
Xfull.drop(['Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday',
       'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7',
       'month_8', 'month_9', 'month_10', 'month_11', 'month_12'], axis=1, inplace = True)


# In[23]:


del Xfull['year']


# ### Remove spatial features

# In[24]:


spatial_cols = ['betweenness', 'lights',  'memorials', 'trees','bus-stops', 'tram-stops', 'metro-stations', 
            'taxi-ranks', 'big-car-parks', 'street_inf_Bicycle Rails', 'street_inf_Bollard','street_inf_Drinking Fountain',
            'street_inf_Floral Crate/Planter Box','street_inf_Horse Trough', 'street_inf_Information Pillar',
            'street_inf_Litter Bin', 'street_inf_Seat', 'street_inf_Tree Guard','landmarks_Community Use', 
            'landmarks_Mixed Use','landmarks_Place Of Assembly', 'landmarks_Place of Worship', 'landmarks_Retail', 
            'landmarks_Transport', 'landmarks_Education Centre','landmarks_Leisure/Recreation', 'landmarks_Office',
       'street_inf_Barbeque', 'street_inf_Hoop', 'street_inf_Picnic Setting', 'landmarks_Specialist Residential Accommodation',
       'landmarks_Vacant Land', 'landmarks_Purpose Built','landmarks_Health Services', 'avg_n_floors', 'buildings_Community Use',
       'buildings_Education', 'buildings_Entertainment', 'buildings_Events','buildings_Hospital/Clinic', 'buildings_Office', 'buildings_Parking',
       'buildings_Public Display Area', 'buildings_Residential','buildings_Retail', 'buildings_Storage', 'buildings_Unoccupied',
       'buildings_Working', 'buildings_Transport']


# In[25]:


columns_to_save = Xfull[spatial_cols]


# In[26]:


Xfull.drop(spatial_cols, axis=1, inplace = True)


# In[27]:


# Keep only the sensor ID
Xfull_sensorid = Xfull.loc[:, Xfull.columns != 'distance_from_centre']
# Keep only the distance from the centre
Xfull_distance_from_centre = Xfull.loc[:, Xfull.columns != 'sensor_id']
# Keep no spatial variables
Xfull_nospatialvariables = Xfull.loc[:, ~Xfull.columns.isin(['sensor_id', 'distance_from_centre'])]
# Version with spatial variables
Xfull = pd.concat([Xfull_distance_from_centre, columns_to_save], axis=1)


# In[30]:


# Dataframe to store the scores for all the models
error_metric_scores = pd.DataFrame()

Xfulls = [Xfull,Xfull_nospatialvariables, Xfull_sensorid, Xfull_distance_from_centre]


# In[31]:


# Dataframe to store the scores for all the models
error_metric_scores = pd.DataFrame()

Xfulls = [Xfull] # Xfull ,Xfull_nospatialvariables, Xfull_sensorid, Xfull_distance_from_centre]
version = ['Original'] #,'Original', No Spatial Features', 'Sensor ID', 'Distance From Centre']
for num in range(0,len(Xfulls)):
    # Get the right Xfull from list
    Xfull=Xfulls[num]
    print(version[num])
    # Use cross_validate to return the error scores associated with this model and this data
    start = time()
    model_output = cross_validate(rf_model_pipeline, Xfull, Yfull, cv=cv_parameters, scoring=error_metrics, error_score="raise")
    end = time()
    print('Ran in {} minutes'.format(round((end - start)/60),2))

    # Formulate the different error scores into a dataframe
    error_metrics_df =pd.DataFrame({'mae': round(abs(model_output['test_neg_mean_absolute_error'].mean()),2), 
                  'mape': round(abs(model_output['test_neg_mean_absolute_percentage_error'].mean()),2),
                  'r2': round(abs(model_output['test_r2'].mean()),2), 
                  'rmse': round(abs(model_output['test_neg_root_mean_squared_error'].mean()),2)},
                 index =[version[num]])

    # Add evaluation metric scores for this model to the dataframe containing the metrics for each model
    error_metric_scores = error_metric_scores.append(error_metrics_df)

    # Save error scores for this distance to file
    #error_metrics_df.to_csv('Results/CV/ComparingModels/{}_{}m_error_metric_scores.csv'.format(model_name,buffer_size_m),index=False)    

    # Save dataframes of error metrics for each buffer distance 
    error_metric_scores.to_csv(f'../Results/CV/ComparingSpatialFeatures/comparingmodels_error_metric_scores_{version}.csv')   


# In[ ]:





# In[ ]:


error_metric_scores

