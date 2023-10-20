## Table of contents

1. [ Cross Validation. ](#crossval)
2. [ Evaluating model performance ](#evalmodelperformance)
3. [ Feature Importance ](#featureimportance)

<a name="crossval"></a>
# Cross validation

Machine learning methods are used to try and model the unknown. The model will never be perfect. The aim is to produce a model which performs well when making predictions on new data. To train the model, you feed data, containing both the target and predictor variables, into the machine learning algorithm and it identifys patterns and determines how to best predict the target variable. You could test how good the method is that the model comes up with for predicting the target variable using the same data used to train it; however, this would likely lead to an overly optimistic view on how the model will generalise to new data (it has already seen the training data and so knows how to handle it). 

The ability to generalise to new data is what we are really interested in, and so model evaluation really requires a seperate test data set not included in the training data fed to the model. In the absence of new data, to estimate a model's skill on new data, we have to use statistical tricks. For instance, fitting the model on a portion of the data and 'holding out' another portion of data, which the model does not see during training, to test the model performance. During this process it is essential to avoid data leakage (see below).

However, by chance the held-out data might be particularly easy or particularly hard to predict. Resampling is an approach to accounting for this. K-fold cross-validation (CV) is an example of resampling. By repeatedly sampling the data into training and hold-out testing sets we are able to account for the model sensitivity to the data it has been trained on. By averaging the performance metrics over many folds we get a better sense of how a model will, on average, generalise to new data. Resampling can thus be used to compare different models. Once the resampling process has been used to estimate the skill of the models under consideration, the resampling procedure is then finished with. 

The purpose of K-Fold CV is not to come up with a final model to make real predictions; rather, to compare the performance of various different models. In this context, model refers to a particular method for describing how some input data relates to what we're trying to predict. All of the models trained during K-fold CV should be thrown away. Then, a final version of the model, which the evaluation showed to best generalise to new data, is fitted on ALL of the data. You could theoretically keep one of the models from the CV process (e.g. if training the models was very computationally expensive). However, the model is likely to perform better when fitted on all available data. 

#### Do we need to evaluate the performance of the final model against a hold out dataset?
K-fold CV provides a reliable/robust evaluation of the model performance. When using k-fold cross-validation, the model is trained and evaluated multiple times on different subsets of the data, which provides a robust estimate of the model's performance. The average performance across all folds can account for the variability in the data and provide a more accurate estimate of the model's generalization ability to new, unseen data.

However, some people still prefer to use a separate holdout set (which is separated off at the very beginning and not used in K-fold CV nor in fitting the final model) for a final evaluation of the chosen model after model selection. However, evaluating the model on a holdout test set only provides a single evaluation of the model, which may be influenced by the specific subset of data used as the holdout set. If the holdout set is not representative of the overall distribution of the data, the evaluation on the holdout set may not be a reliable indicator of the model's performance.

Using a hold-out set might be useful when the data is limited. However, using a hold-out set also limits the amount of data available to the K-fold CV process. 
, to ensure that the model has not overfitted to the training data.

### Data leakage
This occurs when information about the holdout data set (e.g. test or validation data) is made available to the model in the training dataset. This is information that would not be available to the model normally at prediction time. This generally results in  an overly optimistic performance estimate in training (/ a poorer than expected performance with novel data). The data scaling process can introduce data leakage if it is performed before the data is split into train and testing sets. This can be avoided using a Pipeline. 

Time series data -> we cannot choose random samples and assign them to either the test set or the train set because it doesn't make sense to use values from the future to forecast the past. There is a temporal dependency between observations, and this must be preserved during testing.

So, need to ensure that the test set always has a later time stamp than the training set. 

Blocked CV - 

<a name="evalmodelperformance"></a>
# Evaluating regression model performance

Regression predictive modelling involves approximating a mapping function (f) from input variables (X) to a continuous output variable (y) (in contrast, classification involves predicting a category or class label).

Regression model performance <i> cannot </i> be evaluated with an accuracy metric. Accuracy metrics relate to the regularity with which a model predicts exactly the right value. This makes sense for classification problems, but if you are predicting a continuous numeric value then we don’t want to know if the model predicted the value exactly (this might be intractably difficult in practice). Instead, we want to know how close the predictions were to the expected values.

Commonly used error metrics for regression models include:
* Mean Squared Error (MSE).
* Root Mean Squared Error (RMSE).
* Mean Absolute Error (MAE)

### Mean Absoloute Error
Changes in MAE are linear (they do not punish large errors more than small errors).
A perfect MAE would be 0 (so all predictions matched expected values perfectly).
A good MAE is relative to your specific dataset (and its unit is the unit of the dataset).
E.g. A MAE of 10 for the footfall dataset would mean that on average the predicted values are either 10 people higher or lower than the real observed value 

<a name="featureimportance"></a>
# Feature importance
Allows you to understand relationship between the features and the target variable.
By calculating scores for each feature, you can determine which features attribute the most to the predictive power of the model (however, can't say these features are most important for driving the variable of interest)
"The most important feature in predicting the target variable"
The more accurate the model, the more we can trust the importance measures. 
Warning Features that are deemed of low importance for a bad model (low cross-validation score) could be very important for a good model. Therefore it is always important to evaluate the predictive power of a model using a held-out set (or better with cross-validation) prior to computing importances. Permutation importance does not reflect to the intrinsic predictive value of a feature by itself but how important this feature is for a particular model.
 If your model does not generalize accurately, feature importances are worthless. If your model is weak, you will notice that the feature importances fluctuate dramatically from run to run. 
Prior to inspecting feature importances, need to check the model performance. If the model isn't performing well, then looking at the feature importances is basically useless. 

## Methods for feature importance
### Gini importance or Mean Decrease in IMpurity (MDI)
Gini feature importance (or mean decrease in impurity (MDI) counts the number of times a feature is used to split a node, weighted by the number of samples it splits. The mean decrease in impurity importance of a feature is computed by measuring how effective the feature is at reducing variance (regressors) when creating decision trees within RFs.
A RF is a set of decision trees. Each decision tree is an internal set of nodes and leaves. In the internal node the selected feature is used to make a decision on how to divide the data set into two separate sets with similar responses within. The features for internal nodes need a selection criteria. 

Problems with impurity based feature importances:
* They are biased towards high cardinality features (e.g. those which take lots of different values). This can inflate the importance of 
numeric features. This means that if we include an entirely non predictive random variable, containing random numbers, then this may still be ranked quite highly.
* They are based on the training dataset and so importances can even be high for features that are not predictive of the target varaible, as long as the model has the capacity to use them to overfit.

### Permutation feature importance or Mean Decrease in Accuracy (MDA)
The method randomly shuffles each feature and computes the change in model performance using the shuffled feature. The features which impact the model performance the most is the most important one. 
Can be applied to every machine learning model

#### Comparison of Gini and Permutation
Furthermore, impurity-based feature importance for trees are strongly biased and favor high cardinality features (typically numerical features) over low cardinality features such as binary features or categorical variables with a small number of possible categories.
Permutation-based feature importances do not exhibit such a bias. Additionally, the permutation feature importance may be computed performance metric on the model predictions and can be used to analyze any model class (not just tree-based models).
Permutation importance should be quicker because it doesn't require retraining the model?

Good article: https://explained.ai/rf-importance/ 

### Correlated features
If all features are totally independent and not correlated in any way, than computing feature importance individually is no problem. If, however, two or more features are collinear (correlated in some way but not necessarily with a strictly linear relationship) computing feature importance individually can give unexpected results.
Permutation importance gini impportance computed on RFs spreads importance across collinear variables. But this is COLLINEAR and not just correlated.




