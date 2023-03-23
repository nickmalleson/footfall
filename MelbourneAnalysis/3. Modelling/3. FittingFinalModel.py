import copy
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, mean_squared_error,r2_score, accuracy_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import MinMaxScaler
import time as thetime
from sklearn.model_selection import cross_validate
from xgboost import XGBClassifier, XGBRegressor
from time import time
from sklearn.inspection import permutation_importance
from scipy import stats
import math
import pickle
import joblib

from eli5.sklearn import PermutationImportance
from sklearn.model_selection import cross_val_predict

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import multiprocessing

# To display tables in HTML output
from IPython.display import HTML, display

from Functions import *


# ## Read in formatted data
data = pd.read_csv("../Cleaned_data/formatted_data_for_modelling_allsensors_combined_features.csv", index_col = False)

# ### Delete unneeded columns
# We currently include data from all sensors (even incomplete ones)
data = data.drop(['sensor_id'],axis=1) # don't want this included
# Get rid of columns in which none of the sensors have a value
for column in data.columns:
    if np.nanmax(data[column]) ==0:
        del data[column]

# Filter columns using the regex pattern in function input
regex_pattern = 'buildings$|furniture$|landmarks$'
data = data[data.columns.drop(list(data.filter(regex=regex_pattern)))].copy()

# ### Add a random variable (to compare performance of other variables against)
rng = np.random.RandomState(seed=42)
data['random'] = np.random.random(size=len(data))
data["random_cat"] = rng.randint(3, size=data.shape[0])

# ## Prepare data for modelling 
# ### Split into predictor/predictand variables
# The predictor variables
Xfull = data.drop(['hourly_counts'], axis =1)
# The variable to be predicted
Yfull = data['hourly_counts'].values
print("prepared data")

# ### Store the (non Sin/Cos) time columns and then remove them
# Need them later to segment the results by hour of the day
data_time_columns = Xfull[['day_of_month_num', 'time', 'weekday_num', 'time_of_day']]
Xfull = Xfull.drop(['day_of_month_num', 'time', 'weekday_num', 'time_of_day','year', 'month','day', 'datetime', 'month_num'],axis=1)
print(Xfull.columns)
print(len(Xfull.columns))

# Random Forest was the best performing model
# ## Fit the final model
# # 1
# print("fitting model 1")
# rf_model_pipeline1 = Pipeline(steps=[['scaler',StandardScaler()],
#                                     ['rf_regressor',RandomForestRegressor(random_state = 1, n_jobs = 32)]])
# rf_model_pipeline1.fit(Xfull, Yfull)
# print("saving pickled file")
# # Save to pickled file
# filename = 'PickleFiles/rf_model_pipeline1_combined_features.fit.sav'
# joblib.dump(rf_model_pipeline1, filename)

# # 2
# print("fitting model 2")
# rf_model_pipeline2 = Pipeline(steps=[['scaler',StandardScaler()],
#                                     ['rf_regressor',RandomForestRegressor(random_state = 2, n_jobs = 32)]])
# rf_model_pipeline2.fit(Xfull, Yfull)
# print(len(rf_model_pipeline2['rf_regressor'].feature_importances_))
# print("saving pickled file")
# # Save to pickled file
# filename = 'PickleFiles/rf_model_pipeline2_combined_features.fit.sav'
# joblib.dump(rf_model_pipeline2, filename)

# 3
print("fitting model 3")
rf_model_pipeline3 = Pipeline(steps=[['scaler',StandardScaler()],
                                    ['rf_regressor',RandomForestRegressor(n_jobs = 32)]])
rf_model_pipeline3.fit(Xfull, Yfull)
print("saving pickled file")
# Save to pickled file
filename = 'PickleFiles/rf_model_pipeline3_combined_features.fit.sav'
joblib.dump(rf_model_pipeline3, filename)

