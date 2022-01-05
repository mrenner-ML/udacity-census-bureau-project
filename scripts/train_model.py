'''
Script to train and evaluate machine learning model.
'''

import argparse
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model,inference,compute_model_metrics

parser = argparse.ArgumentParser(description='Pass input data for training.')
parser.add_argument('--input_data', help='input data for training model')

args = parser.parse_args()

data = pd.read_csv(args.input_data)

def slice_eval(cat_features,X_test_orig,y_test,preds):
    # instantiate lists where results will be stored
    feat_ls,cat_ls,precision_ls,recall_ls,fbeta_ls = [],[],[],[],[]
    # run evaluation of overall test dataset
    precision, recall, fbeta = compute_model_metrics(
                y_test,preds)
    for ls,name in zip(
                [feat_ls, cat_ls, precision_ls, recall_ls, fbeta_ls],
                ['all', 'all', precision, recall, fbeta]):
                ls.append(name)
    # run evaluation for each class of each categorical features
    for feat in cat_features:
        for cat in X_test_orig[feat].unique():
            mask = (X_test_orig[feat] == cat)
            precision, recall, fbeta = compute_model_metrics(
                y_test[mask],preds[mask])
            for ls,name in zip(
                [feat_ls, cat_ls, precision_ls, recall_ls, fbeta_ls],
                [feat, cat, precision, recall, fbeta]):
                ls.append(name)
    return feat_ls, cat_ls, precision_ls, recall_ls, fbeta_ls

def main():
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    print('Processing training data')
    X_orig, X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )

    print('Spliting data')
    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=0.20,random_state=0)
    _, X_test_orig, _, _ = train_test_split(
        X_orig,y, test_size=0.20,random_state=0)
    
    print('Training model')
    clf = train_model(X_train,y_train)

    print('Saving model and encoder')
    pickle.dump(clf, open('model.pkl', 'wb'))
    pickle.dump(encoder, open('encoder.pkl', 'wb'))

    print('Evaluating model')
    pickled_model = pickle.load(open('model.pkl', 'rb'))
    preds = inference(pickled_model,X_test)
    feat_ls, cat_ls, precision_ls, recall_ls, fbeta_ls = slice_eval(
        cat_features,X_test_orig,y_test,preds)
    
    print('Exporting results')
    pd.DataFrame({
        'variable':feat_ls,
        'category':cat_ls,
        'precision':precision_ls,
        'recall':recall_ls,
        'fbeta':fbeta_ls
    }).to_csv('slice_output.txt',index=False)

if __name__ == "__main__":
    main()
