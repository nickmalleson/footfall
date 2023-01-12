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
