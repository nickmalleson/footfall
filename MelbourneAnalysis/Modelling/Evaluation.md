### Feature importance
Allows you to understand relationship between the features and the target variable.
By calculating scores for each feature, you can determine which features attribute the most to the predictive power of the model (however, can't say these features are most important for driving the variable of interest)
"The most important feature in predicting the target variable"
The more accurate the model, the more we can trust the importance measures. 
Warning Features that are deemed of low importance for a bad model (low cross-validation score) could be very important for a good model. Therefore it is always important to evaluate the predictive power of a model using a held-out set (or better with cross-validation) prior to computing importances. Permutation importance does not reflect to the intrinsic predictive value of a feature by itself but how important this feature is for a particular model.
 If your model does not generalize accurately, feature importances are worthless. If your model is weak, you will notice that the feature importances fluctuate dramatically from run to run. 

#### Methods for feature importance
##### Gini importance or Mean Decrease in IMpurity (MDI)
A RF is a set of decision trees. Each decision tree is an internal set of nodes and leaves. In the internal node the selected feature is used to make a decision on how to divide the data set into two separate sets with similar responses within. The features for internal nodes need a selection criteria. 
##### Permutation feature importance or Mean Decrease in Accuracy (MDA)
 the method randomly shuffles each feature and computes the change in model performance using the shuffled feature. The features which impact the model performance the most is the most important one. 
Can be applied to every machine learning model

. The mean decrease in impurity importance of a feature is computed by measuring how effective the feature is at reducing variance (regressors) when creating decision trees within RFs.

Can apply with eli5 and from sklearn.inspection import permutation_importance

#### Comparison of Gini and Permutation
Furthermore, impurity-based feature importance for trees are strongly biased and favor high cardinality features (typically numerical features) over low cardinality features such as binary features or categorical variables with a small number of possible categories.
Permutation-based feature importances do not exhibit such a bias. Additionally, the permutation feature importance may be computed performance metric on the model predictions and can be used to analyze any model class (not just tree-based models).
Permutation importance should be quicker because it doesn't require retraining the model?

Good article: https://explained.ai/rf-importance/ 

### Correlated features
If all features are totally independent and not correlated in any way, than computing feature importance individually is no problem. If, however, two or more features are collinear (correlated in some way but not necessarily with a strictly linear relationship) computing feature importance individually can give unexpected results.
Permutation importance gini impportance computed on RFs spreads importance across collinear variables. But this is COLLINEAR and not just correlated.


