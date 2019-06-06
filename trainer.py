import model_utils
import numpy
import tqdm


def try_decision_tree(training,validation):
    print('Creating Decision Tree Model...')
    (t_features,t_labels) = training
    (v_features,v_labels) = validation

    from sklearn.tree import DecisionTreeClassifier

    def makeTree(features,labels,d):
        clf_ent = DecisionTreeClassifier(criterion="entropy",max_depth=d)
        clf_ent.fit(features,labels)
        return clf_ent
    
    mindepth,maxdepth = 1,12
    print("Trying trees from depth {} to {}".format(mindepth,maxdepth))

    trees = []
    for d in range(mindepth,maxdepth+1):
        tree = makeTree(t_features,t_labels,d)

        t_pred = tree.predict(t_features)
        t_acc = sum([1 if(t_labels[i]==p) else 0 for i,p in enumerate(t_pred)])/len(t_labels)
        v_pred = tree.predict(v_features)
        v_acc = sum([1 if(v_labels[i]==p) else 0 for i,p in enumerate(v_pred)])/len(v_labels)
        print('Decision Tree: Accuracy with d={}: (t:{:.4f}), (v:{:.4f})'.format(d, t_acc, v_acc))
        trees.append((v_acc,tree))

    trees.sort(key=lambda x: x[0],reverse=True)

    (_,model) = trees[0]
    return model


def try_rand_forest(training,validation):
    print('Creating Random Forest Model...')
    (t_features,t_labels) = training
    (v_features,v_labels) = validation

    from sklearn.ensemble import RandomForestClassifier

    def makeForest(features,labels):
        forest = RandomForestClassifier(n_estimators=50,criterion='gini',max_depth=8)
        forest.fit(features,labels)
        return forest

    
    attempts = 5
    print("Attempting {} times".format(attempts))

    forests = []
    for i in range(attempts):
        forest = makeForest(t_features,t_labels)
        t_pred = forest.predict(t_features)
        t_acc = sum([1 if(t_labels[i]==p) else 0 for i,p in enumerate(t_pred)])/len(t_labels)
        v_pred = forest.predict(v_features)
        v_acc = sum([1 if(v_labels[i]==p) else 0 for i,p in enumerate(v_pred)])/len(v_labels)
        print('Random Forest (Attempt {}): (t:{:.4f}), (v:{:.4f})'.format(i,t_acc, v_acc))
        forests.append((v_acc,forest))

    forests.sort(key=lambda x:x[0], reverse=True)
    (_,model) = forests[0]

    return model


def try_log_reg(training,validation):
    print('Creating Logistic Regression Model...')
    (t_features,t_labels) = training
    (v_features,v_labels) = validation

    from sklearn.linear_model import LogisticRegression

    model_utils.normalize_features(t_features)
    model_utils.normalize_features(v_features)

    attempts = 5
    print("Attempting {} times".format(attempts))
    
    models = []
    for i in range(attempts):
        logreg = LogisticRegression(penalty='l2',solver='newton-cg',max_iter=200,fit_intercept=True)
        logreg.fit(t_features,t_labels)

        t_pred = logreg.predict(t_features)
        t_acc = sum([1 if(t_labels[i]==p) else 0 for i,p in enumerate(t_pred)])/len(t_labels)
        v_pred = logreg.predict(v_features)
        v_acc = sum([1 if(v_labels[i]==p) else 0 for i,p in enumerate(v_pred)])/len(v_labels)
        print('Logistic Regression (Attempt {}): Accuracy: (t:{:.4f}), (v:{:.4f})'.format(i,t_acc, v_acc))
        models.append((v_acc,logreg))
    
    models.sort(key=lambda x:x[0],reverse=True)
    (_,model) = models[0]

    return model


def test_data_validity(training):
    (features,labels) = training
    import math
    clean = True
    for inst_idx in tqdm.trange(features.shape[0]):
        for feature_idx in range(features.shape[1]):
            if(math.isnan(features[inst_idx,feature_idx])):
                print("NaN Error: Feature {} of instance {} = {}".format(feature_idx,inst_idx,features[inst_idx,feature_idx]))
                clean=False
                break
    if(clean):
        print("Data is clean of errors!")


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('training_data',help='Training data file')
    argparser.add_argument('output_model',help='Output for model location')
    argparser.add_argument('model_type',help='Choose model type: tree, forest, logreg')
    args = argparser.parse_args()
    training_filename = args.training_data
    training, validation = model_utils.read(training_filename, validation=.2)
    print('Training: {} labels, {} features\nValidation: {} labels, {} features'.format(training[0].shape[0],training[0].shape[1],validation[0].shape[0],validation[0].shape[1]))

    test_data_validity(training)

    model = None
    if(args.model_type == 'tree'):
        model = try_decision_tree(training,validation)
    elif(args.model_type == 'forest'):
        model = try_rand_forest(training,validation)
    elif(args.model_type == 'logreg'):
        model = try_log_reg(training,validation)
    else:
        print('Unrecognized model type! Must be "tree", "forest", or "logreg"')

    model_utils.dump_model(model,args.output_model)
    #predict_test(model,args.testing_data,args.output_file)


