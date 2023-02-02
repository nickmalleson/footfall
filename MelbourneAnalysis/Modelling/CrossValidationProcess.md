### Cross validation process
Using cross validation to compare the performance of different models is useful because models are sensitive to the data they are trained on. By averaging the performance metric over many folds we can get some sense of which model will, on average, best generalise to new data. 

Model selection:  There are metrics which we can use to evaluate how efficient a model is. But if we do this on just one train-test split then how can we be sure it is an accurate evaluation and not too low or too high by chance? So if we are comparing between two models, decision about which one is better might also be based on chance. 

1. training dataset is used to train a few candidate models
2.  validation dataset is used to evaluate the candidate models
3. one of the candidates is chosen
4. the chosen model is trained with a new training dataset
5. the trained model is evaluated with the test dataset




THe dataset we used in the final evaluiation must be different to that used in cross-validation because we don't want data leakage (but if its a new model, surely it doesn't matter?)
Out of sample validation: 


Divide traning set into several validation sets (folds, in the jargon), and see how each model performs for the validation set based on the rest of the training data.

Model evaluation is a method of assessing the correctness of models on test data. The test data consists of data points that have not been seen by the model before.
Out of sample predictons - predicitons made by a model on data not used during the training of the model. 
PRedictions made on data not used to train a model provide insight into how the model will generalise to new situations. 

So, you could also fit a model on the training data, and then use it to predict the training data. If the score is better for predict the training data than the test data, then the model is overfitting. 

https://machinelearningmastery.com/train-final-machine-learning-model/ 
### How to fit a final predictive model after K-fold validation?
CV allows us to conclude whether one model is better than another (NB: in this context 'model' refers to a particular method for describing how some input data relates to what we are trying to predict. We don't generally refer to particular instances of methods as different models. However, the purpose of CV is not to come up with a final model to use to make real predictions. For that, we want to use all the data we have to come up with the best model possible. 

You finalise a model by applying the chosen machine learning procedure on all of your data.

Why not keep the best model from cross validation? You can if you like. You may save time and effort by reusing one of the models trained during skill estimation. This can be a big deal if it takes days, weeks, or months to train a model. Your model will likely perform better when trained on all of the available data than just the subset used to estimate the performance of the model. This is why we prefer to train the final model on all available data.

If you train a model on all the available data, then how do you know how well the model will perform? You have already answered this in the K-fold validation (resampling) process. 

What is the point of keeping back a test data set to test the final model? What if the skill is poor on this test set? It's redundant and could be misleading. If it is then: 1. This 5% is a sample that is not representative of data . i.e.. Occurred by chance. So i should have other approach to test on representative of the data. 2. Model is not good enough or over-fitted – Even this time i cannot come to conclusion as 5% sample may not be representative of data.

Model performance scores - averaged across CV runs. 


Applied machine elarning is concerned with more than performing a good performing model; it also requires finding an appropriate sequence of data preperation steps and steps for the post processing of predictions. 

### Data leakage
This occurs when information about the holdout data set (e.g. test or validation data) is made available to the model in the training dataset. This is information that would not be available to the model normally at prediction time. This generally results in  an overly optimistic performance estimate in training (/ a poorer than expected performance with novel data). The data scaling process can introduce data leakage if it is performed before the data is split into train and testing sets. This can be avoided using a Pipeline. 

