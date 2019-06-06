import model_utils
import numpy


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('testing_data',help='Training data file')
    argparser.add_argument('input_model',help='Output for model location')
    argparser.add_argument('output_file',help='File to output predictions')
    argparser.add_argument('--scale',help='Makes the feature space normalized. Good for LogReg',action='store_true')
    args = argparser.parse_args()

    testing = model_utils.read(args.testing_data,testing=True)
    print("Testing on {} labels...".format(testing[0].shape[0]))


    (features,ids) = testing
    model = model_utils.load_model(args.input_model)
    
    if(args.scale):
        features = model_utils.normalize_features(features)

    predictions = model.predict(features)
    with open(args.output_file,'w+') as f:
        for i,prediction in enumerate(predictions):
            f.write('{},{}\n'.format(ids[i],prediction))

