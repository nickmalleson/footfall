### Gini feature importance
Gini feature importance (or mean decrease in impurity (MDI) counts the number of times a feature is used to split a node, weighted by the number of samples it splits

https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html  

Problems with impurity based feature importances:
* They are biased towards high cardinality features (e.g. those which take lots of different values). This can inflate the importance of 
numeric features. This means that if we include an entirely non predictive random variable, containing random numbers, then this may still be ranked quite highly.
* They are based on the training dataset and so importances can even be high for features that are not predictive of the target varaible, as long as the model has the capacity to use them to overfit.
Prior to inspecting feature importances, need to check the model performance. If the model isn't performing well, then looking at the feature importances is basically useless. 
