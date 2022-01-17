'''
Module for creating API that processes the data and runs inference
'''

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
    age: int = 39
    workclass: str = "State-gov"
    fnlgt: int = 77516
    education: str = "Bachelors"
    education_num: int = Field(13, alias='education-num')
    marital_status: str = Field("Never-married",alias='marital-status')
    occupation: str = "Adm-clerical"
    relationship: str = "Not-in-family"
    race: str = "White"
    sex: str = "Male"
    capital_grain: int = Field(2174, alias='capital-gain')
    capital_loss: int = Field(0, alias='capital-loss')
    hours_per_week: int = Field(40, alias='hours-per-week')
    native_country: str = Field("United-States",alias='native-country')

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
    return 'HELLO!'

@app.post("/predict")
async def run_inference(input_data: InputData):
    data = pd.DataFrame(input_data.dict(),index=[0])
    X = process_data(data,cat_features,encoder)
    pred = model.predict_proba(X)
    proba_low_salary,proba_high_salary = pred[0,0], pred[0,1]
    return {"proba_low_salary":proba_low_salary,
            "proba_high_salary":proba_high_salary}
