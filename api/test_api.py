from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

def test_welcome_message():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == 'HELLO!'

def test_correct_input():
    response = client.post(
        "/predict",
        json={
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
        },
    )
    assert response.status_code == 200
    content = response.json()
    assert (content['proba_low_salary'] + content['proba_high_salary']) == 1

def test_incorrect_input():
    response = client.post(
        "/predict",
        json={
            "age": "WRONG_STRING",
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
        },
    )
    assert response.status_code != 200