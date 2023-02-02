### Cross validation process
Model selection:  There are metrics which we can use to evaluate how efficient a model is. But if we do this on just one train-test split then how can we be sure it is an accurate evaluation and not too low or too high by chance? So if we are comparing between two models, decision about which one is better might also be based on chance. 

1. training dataset is used to train a few candidate models
2.  validation dataset is used to evaluate the candidate models
3. one of the candidates is chosen
4. the chosen model is trained with a new training dataset
5. the trained model is evaluated with the test dataset

CV allows us to conclude whether one model is better than another. 
THe dataset we used in the final evaluiation must be different to that used in cross-validation because we don't want data leakage (but if its a new model, surely it doesn't matter?)
Out of sample validation: 


Divide traning set into several validation sets (folds, in the jargon), and see how each model performs for the validation set based on the rest of the training data.

Model evaluation is a method of assessing the correctness of models on test data. The test data consists of data points that have not been seen by the model before.
Out of sample predictons - predicitons made by a model on data not used during the training of the model. 
PRedictions made on data not used to train a model provide insight into how the model will generalise to new situations. 

So, you could also fit a model on the training data, and then use it to predict the training data. If the score is better for predict the training data than the test data, then the model is overfitting. 
