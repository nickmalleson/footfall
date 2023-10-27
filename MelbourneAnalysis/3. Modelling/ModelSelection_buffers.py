#!/usr/bin/env python
# coding: utf-8

# # Model selection
# 
# Cross-validation is used here to select the best model. In this script it is used to test the best buffer size to draw around the sensors from within which to draw the environment variables. 
# 
# Tests the performance of a <u>Random Forest Regressor</u>
# 
# <u> The following variables are included in the model:</u>
# * Weather variables (rain, temperature, windspeed)
# * Time variables (Day of week, month, year, time of day, public holiday)
# * Sensor environment variables:
#     * Betweenness of the street 
#     * Buildings in proximity to the sensor
#     * Landmarks in proximity to the sensor  
#     * Furniture in proximity to the sensor    
#     * Lights in proximity to the sensor   
# 
# 
# <u> Model performance is evaluated for a range of buffer sizes around the sensors within which the environment variables are counted</u>:
#    * 50
#    * 100
#    * 200
#    * 400
#    * 500
#    * 600
#    * 1000

# In[1]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor 
import xgboost as xgb
from sklearn.pipeline import Pipeline
import folium
import branca.colormap as cm
from eli5.sklearn import PermutationImportance
import joblib
import os
import psutil

from Functions import *


# ## Run models with cross-validation

# ### Define the error metrics for the cross-validation to return, and the parameters of the cross validation

# In[2]:


error_metrics = ['neg_mean_absolute_error', 'r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_percentage_error']
cv_parameters = KFold(n_splits=10, random_state=1, shuffle=True)


# ### Use CV to return error metrics for the datasets produced with different buffer sizes

# In[ ]:


# Dataframe to store the scores for all the models
error_metric_scores = pd.DataFrame()

# Set up model pipeline
model = Pipeline(steps=[['scaler',StandardScaler()],['rf_regressor', RandomForestRegressor(random_state = 1, n_jobs = 10)]])

# Define parameters
model_name = 'rf_regressor'

# Loop through each buffer size option
for buffer_size_m in [50,100,200, 400,500,600,1000]:
    print(buffer_size_m)
    # Read in data
    input_csv ="../Cleaned_data/FormattedDataForModelling/formatted_data_for_modelling_allsensors_{}.csv".format(buffer_size_m)
    Xfull, Yfull,data_time_columns = prepare_x_y_data(input_csv)
    
    # Remove year
    del Xfull['year']
    
    # Cut off data post-Covid
    Xfull= Xfull[0:2643750]
    Yfull= Yfull[0:2643750]
    data_time_columns = data_time_columns[0:2643750] # end of 2019

    # Drop dummy varables
    Xfull.drop(['Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday',
       'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7',
       'month_8', 'month_9', 'month_10', 'month_11', 'month_12'], axis=1, inplace = True)
    
    # Use cross_validate to return the error scores associated with this model and this data
    start = time()
    model_output = cross_validate(model, Xfull, Yfull, cv=cv_parameters, scoring=error_metrics, error_score="raise")
    end = time()
    print('Ran in {} minutes'.format(round((end - start)/60),2))
    
    # Formulate the different error scores into a dataframe
    error_metrics_df =pd.DataFrame({'mae': round(abs(model_output['test_neg_mean_absolute_error'].mean()),2), 
                  'mape': round(abs(model_output['test_neg_mean_absolute_percentage_error'].mean()),2),
                  'r2': round(abs(model_output['test_r2'].mean()),2), 
                  'rmse': round(abs(model_output['test_neg_root_mean_squared_error'].mean()),2)},
                 index =["{}".format(buffer_size_m)])
        
    # Add evaluation metric scores for this model to the dataframe containing the metrics for each model
    error_metric_scores = error_metric_scores.append(error_metrics_df)
    # Save error scores for this distance to file
    error_metrics_df.to_csv('Results/CV/ComparingBufferSizes/{}/{}_error_metrics_{}m.csv'.format(buffer_size_m, model_name,buffer_size_m),index=False)    

# Save dataframes of error metrics for each buffer distance 
error_metric_scores.to_csv('Results/CV/ComparingBufferSizes/error_metric_scores.csv')   


# ### Print table showing error metrics associated with each buffer size

# In[ ]:


error_metric_scores


# In[ ]:


error_metric_scores = pd.DataFrame()
for buffer_size_m in [400,500,600,1000]:
    csv =pd.read_csv('Results/CV/ComparingBufferSizes/{}/{}_error_metrics_{}m.csv'.format(buffer_size_m, model_name,buffer_size_m))
    csv.insert (0, "buffer_size", buffer_size_m)
    error_metric_scores = error_metric_scores.append(csv)


# In[ ]:


error_metric_scores.to_csv('Results/CV/ComparingBufferSizes/error_metric_scores_new.csv')   

