# CS 434: Class Competition, PK Detection in RNA

***Alex Edwards***,
***Jacob Dugan***

There are two primary scripts to generate predictions for a given test file.
```trainer.py``` is the script which will generate a model given some training data.
```tester.py``` will output a file with predicted classes on each instance ID, given some model (made with ```trainer.py```).

### Predictions

The predictions generated for the competition assignment are made using a random forest, decision tree, and logistic regression. The given AUC is on the train

* ```features103_pred1.txt```: Random Forest predictions using 103 features.
* ```features103_pred2.txt```: Decision Tree predictions using 103 features.
* ```features103_pred3.txt```: Logistic Regression predictions using 103 features.
* ```featuresall_pred1.txt```: Random Forest predictions using all 1053 features.
* ```featuresall_pred2.txt```: Decision Tree predictions using all 1053 features.
* ```featuresall_pred3.txt```: Logistic Regression predictions using all 1053 features.

The testing is done on the given ```features103_test.txt``` and ```featuresall_test.txt``` files.

Note that ```featuresall_train.txt``` and ```featuresall_test.txt``` are not included in this repository due to file size limits.

<hr>

### How to use

#### To create a model with training data:

```$ python3 trainer.py <training data> <output destination> <model type>```

* ```<training data>``` is the file which contains the training data, including both features and labels. The first column should be the instance ID, the second column is the class label (0 or 1), and all following columns are training features. A header on this csv file is expected.
* ```<output destination>``` is the name of the file to be created. This file is the model exported by the joblib library in python. 
* ```<model type>``` is a string with the type of model to be created. There are three options:
- ```tree```: A decision tree, with the tree with the best performance on a holdout set returned, with depths 1~12. Based on the scikit-learn library.
- ```forest```: A random forest, based on the scikit-learn library.
- ```logreg```: A logistic regression model, based on the scikit-learn library. This model will normalize the features between 0-1.

#### To test the model on some data:

```$ python3 tester.py <testing file> <input model> <output destination>```

* ```<testing file>```: The file with testing data. The structure of the file is slightly different than the training data. The instance IDs are in the first column. Since there are no labels, the features should follow after the ID, with all columns after the first being a feature. The order of features should be identical to training.
* ```<input model>```: The model we are using, generated from the ```trainer.py``` script.
* ```<output destination>```: The output file destination. This will be in csv format, with the first column being the instance ID and the second column being the predicted class label. This file will be used to score the model.

#### Scoring the model
Use the provided scoring script from the assignment

```$ python3 score.py <prediction> <ground truth>```

* ```<prediction>```: The predictions of the model given from the ```tester.py``` script.
* ```<ground truth>```: The ground truth of the data we tested on. 

#### Other scripts
* ```model_utils.py```: A file containing utility functions used in both training and testing scripts.
* ```quick_check.py```: Gives a quick look at the predictions' class distributions.