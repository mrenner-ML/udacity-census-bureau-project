'''
Module for creating API that processes the data and runs inference
'''
import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

import pandas as pd
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

def process_data(data,cat_features,encoder):
    '''
    Function for processing data for inference
    '''
    X_categorical = data[cat_features].values
    X_continuous = data.drop(*[cat_features], axis=1)
    X_categorical = encoder.transform(X_categorical)
    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X

# coerce data to appropriate type and add aliases
class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_grain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')

    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
        }

# load encoder and model artifact
encoder = pickle.load(open('./scripts/encoder.pkl', 'rb'))
model = pickle.load(open('./scripts/model.pkl', 'rb'))

# define categorical features
cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

# create API
app = FastAPI()

@app.get('/')
async def say_hello():
    return {"message": "Hello Udacity"}

@app.post("/predict")
async def run_inference(input_data: InputData):
    data = pd.DataFrame(input_data.dict(),index=[0])
    X = process_data(data,cat_features,encoder)
    pred = model.predict_proba(X)
    proba_low_salary,proba_high_salary = pred[0,0], pred[0,1]
    return {"proba_low_salary":proba_low_salary,
            "proba_high_salary":proba_high_salary}
