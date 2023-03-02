## Table of contents

1. [ Cross Validation] #crossval

<a name="crossval"></a>
## Cross validation

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