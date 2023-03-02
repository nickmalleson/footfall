# Modelling workflow

### <ins> PrepareDataForModelling.ipynb </ins>

Gets the data into the format required in modelling

###  <ins> TestModelsWithCV.ipynb  </ins>
Tests the performance of a number of different machine learning models using 10-fold cross validation. These include:
* Linear regression
* Random Forest
* XGBoost
* Extra Trees Regressor

The performance of the models is also evaluated in respect to whether certain predictive variables are included in the model or not. The options include:
* Include all the built environment variables (including their subtypes), street betweenness, weather variables and time based variables
* Include all the built environment variables (but just the headline categories, e.g. furniture, buildings, landmarks), street betweenness, weather variables and time based variables
* c Include just weather and time based variables

The outputs of the 10-fold cross validation process are:
* The error metric scores associated with that model (averaged over all folds)
    * The Mean absoloute error and the Mean absoloute percentage error
* The Gini feature importance associated with each fold
* The model used in each fold, from which the permutation imporance associated with each fold can be calculated
* A prediction for each data point in the dataset (within cross validation each data point is included in the test set only once and thus despite their beng multiple cross-validation folds, each true value of Y has only one associated prediction )

### <b> Conclusion: Random Forest Regressor is best performing model </b>

###  <ins> FittingFinalModel.ipynb/py  </ins>
Fit a Random Forest Regressor on the whole dataset.

####  <ins> AssessingFinalmodel.ipynb  </ins>
Find the Gini and Permutation feature importances returned from the final fitted model.  

#### <ins> Make predictions? </ins>
Can also make a script to use the model to predict for new data?
