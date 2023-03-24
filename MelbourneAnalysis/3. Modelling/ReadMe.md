# Modelling workflow

###  <ins> 1. TestingCorrelations.ipynb  </ins>

###  <ins> 2. TestModelsWithCV.ipynb  </ins>
Tests the performance of a number of different machine learning models using 10-fold cross validation. These include:
* Linear regression
* Random Forest
* XGBoost
* Extra Trees Regressor

The performance of the models is also evaluated in respect to whether certain predictive variables are included in the model or not. The options include:
* Include all the built environment variables (including their subtypes), street betweenness, weather variables and time based variables
* Include all the built environment variables (but just the headline categories, e.g. furniture, buildings, landmarks), street betweenness, weather variables and time based variables
* Include just weather and time based variables

The outputs of the 10-fold cross validation process are:
* The error metric scores associated with that model (averaged over all folds)
    * The MAE, the MAPE and the RMSE
* The Gini feature importance associated with each fold
* The model used in each fold, from which the permutation imporance associated with each fold can be calculated
* A prediction for each data point in the dataset (within cross validation each data point is included in the test set only once and thus despite their beng multiple cross-validation folds, each true value of Y has only one associated prediction )

### <b> Conclusion: Random Forest Regressor is best performing model </b>


###  <ins> 2.b. TestModelsWithCV-AggregatedModel.ipynb  </ins>

The same but with the data for each hour across all the sensors aggrgated into just one value for the whole city.

###  <ins> 3. FittingFinalModel.ipynb/py  </ins>
Fit a Random Forest Regressor on the whole dataset. Saves this model to a pickle file so it can be reused in future. Also saves the predictor variables used in constructing it, and the real values associated with those predictor variables.

###  <ins> 4. AssessingFinalmodel.ipynb  </ins>
Reads in the Random Forest model fitted on the whole dataset from the pickle file. 
Find the Gini and Permutation feature importances returned from this final fitted model.  
There are no predicted values to plot, because we fitted the model on the whole dataset.

#### <ins> Make predictions? </ins>
Can also make a script to use the model to predict for new data?
