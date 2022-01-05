from contextlib import nullcontext
import pandas as pd
import pickle
import pytest

from scripts.ml.data import process_data
from scripts.ml.model import inference, compute_model_metrics


@pytest.fixture
def data():
    '''fixture to read data'''
    df = pd.read_csv('../data/cleaned_census.csv')
    return df

@pytest.fixture
def model():
    '''fixture to read model'''
    pickled_model = pickle.load(open('model.pkl', 'rb'))
    return pickled_model

@pytest.fixture
def categorical_features():
    '''fixture for categorical variables'''
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
    return cat_features

def test_processing(data, categorical_features):
    '''testing data processing function'''
    X_orig, X, y, encoder, lb = process_data(data, categorical_features,'salary')
    # check if all columns from X are only numeric
    assert X.shape[0] == y.shape[0]
    assert len(X) == len(X_orig)

def test_inference(model,data,categorical_features):
    '''testing inference function'''
    X_orig, X, y, encoder, lb = process_data(data, categorical_features,'salary')
    preds = inference(model,X)
    assert preds is not None
    assert preds.shape[0]==X.shape[0]

def test_eval(model,data,categorical_features):
    '''testing evaluation function'''
    X_orig, X, y, encoder, lb = process_data(data, categorical_features,'salary')
    preds = inference(model,X)
    precision, recall, fbeta = compute_model_metrics(y,preds)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1

