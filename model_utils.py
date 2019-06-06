import pandas
import numpy

def read(filename,validation=0.,testing=False):
    dataframe = pandas.read_csv(filename,sep='\t',encoding='utf-8')
    dataframe = dataframe.fillna(0.)
    data = dataframe.values
    if(not testing):
        numpy.random.shuffle(data)

    if(testing):
        testing_id = data[:, 0]
        testing_features = data[:, 1:]
        testing_features = testing_features.astype('float64')

        return (testing_features,testing_id)
    else:
        num_for_validation = int(validation * dataframe.shape[0])
        training_features = data[num_for_validation:, 2:]
        validation_features = data[:num_for_validation, 2:]
        training_labels = data[num_for_validation:, 1]
        validation_labels = data[:num_for_validation, 1]
        training_features = training_features.astype('float64')
        validation_features = validation_features.astype('float64')
        training_labels = training_labels.astype('int')
        validation_labels = validation_labels.astype('int')

        if(validation>0):
            return (training_features,training_labels), (validation_features,validation_labels)
        else:
            return (training_features,training_labels)

def read_for_testing(filename):
    dataframe = pandas.read_csv(filename,sep='\t',encoding='utf-8')
    dataframe = dataframe.fillna(0.)
    data = dataframe.values

    testing_id = data[:, 0]
    testing_features = data[:, 1:]
    testing_features = testing_features.astype('float64')

    return (testing_features,testing_id)

def dump_model(model,name):
    from joblib import dump
    dump(model,name)

def load_model(name):
    from joblib import load
    return load(name)

def normalize_features(features):
    from sklearn.preprocessing import StandardScaler
    scalar = StandardScaler()
    scalar.fit(features)
    features = scalar.transform(features)
    return features

class ModelAggregator:
    def __init__(self):
        self.models = []

    def add(self,model):
        self.models.append(model)

    def predict(self,file_predict):
        preds = []
        for model in self.models:
            p = model.predict(file_predict)
            preds.append(p)
        
        probability_predictions = []
        for instance in range(len(preds[0])):
            probability = sum([preds[i][instance] for i in range(len(preds))]) / len(preds)
            probability_predictions.append(probability)

        return probability_predictions
