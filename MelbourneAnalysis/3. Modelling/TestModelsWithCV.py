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

from Functions import *

# ## Read in formatted data
buffer_size_m = 200
data = pd.read_csv("../Cleaned_data/FormattedDataForModelling/formatted_data_for_modelling_allsensors_{}.csv".format(buffer_size_m), index_col = False)
data = data.fillna(0)
print("buffer of: ", buffer_size_m)

# ### Delete unneeded columns
# We currently include data from all sensors (even incomplete ones)
sensor_ids = data['sensor_id']
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


# ### Store the (non Sin/Cos) time columns and then remove them
# Need them later to segment the results by hour of the day

data_time_columns = Xfull[['day_of_month_num', 'time', 'weekday_num', 'time_of_day']]
Xfull = Xfull.drop(['day_of_month_num', 'time', 'weekday_num', 'time_of_day','year', 'month','day', 'datetime', 'month_num'],axis=1)

# ## Define model pipelines (linear regression, random forest and XGBoost)
# Include process to scale the data
lr_model_pipeline = Pipeline(steps=[['scaler',StandardScaler()],['linear_regressor',LinearRegression()]])
rf_model_pipeline = Pipeline(steps=[['scaler',StandardScaler()],['rf_regressor', RandomForestRegressor(random_state = 1, n_jobs = 16)]])
# xgb_model_pipeline = Pipeline(steps=[['scaler',StandardScaler()],['xgb_regressor',xgb.XGBRegressor(random_state=1, n_jobs = 10)]])
# et_model_pipeline = Pipeline(steps=[['scaler',StandardScaler()],['et_regressor',ExtraTreesRegressor (random_state = 1, n_jobs = 10)]])

# ## Run models with cross-validation
# ### Define the error metrics for the cross-validation to return, and the parameters of the cross validatio
error_metrics = ['neg_mean_absolute_error', 'r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_percentage_error']
cv_parameters = KFold(n_splits=10, random_state=1, shuffle=True)


# ### Define regex's to remove columns not needed in various splits of removing column
column_regex_dict = {'withsubtypes':'buildings$|furniture$|landmarks$'}


# ### Loop through each combination of the models, and the variables to include in the modelling
error_metric_scores = pd.DataFrame()


# Dictionary to store dataframes of feature importance scores
predictions_df = pd.DataFrame()

feature_importance_scores ={}


model = rf_model_pipeline
model_name ="rf_regressor"
regex_name = 'with_subtypes'
regex = 'buildings$|furniture$|landmarks$'
    
print("rf_regressor")
# Run the model: return the estimators and a dataframe containing evaluation metrics
estimators, error_metrics_df, feature_list, predictions = run_model_with_cv_and_predict(
    model, model_name, error_metrics, cv_parameters, Xfull, Yfull, regex_name, regex) 
# Add evaluation metric scores for this model to the dataframe containing the metrics for each model
error_metric_scores = error_metric_scores.append(error_metrics_df)

predictions_df[model_name] =predictions

# Create dataframe of feature importances (no feature importances for linear regression)
if model_name != 'linear_regressor':
    feature_importances = pd.DataFrame(index =[feature_list])
    for idx,estimator in enumerate(estimators):
            feature_importances['Estimator{}'.format(idx)] = estimators[idx][model_name].feature_importances_
    feature_importance_scores["{}_{}".format(model_name, regex_name)] = feature_importances

filename = 'PickleFiles/CV/{}/{}_cv_estimators.pkl'.format(buffer_size_m, model_name)
joblib.dump(estimators, filename)

        
feature_importances_df = feature_importance_scores["rf_regressor"].copy()
feature_importances_df.reset_index(inplace=True)
feature_importances_df.rename(columns={'level_0':'Variable'},inplace=True)        
        
error_metric_scores.to_csv('PickleFiles/CV/{}/error_metric_scores.csv'.format(buffer_size_m), index= False)   
predictions_df.to_csv('PickleFiles/CV/{}/predictions'.format(buffer_size_m), index= False)   
feature_importances_df.to_csv('PickleFiles/CV/{}/rf_feature_importances.csv'.format(buffer_size_m), index= False)   

