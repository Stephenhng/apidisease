from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import uvicorn
import json
from starlette.middleware.cors import CORSMiddleware


app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Symptom(BaseModel):
    symptom1 : float
    symptom2 : float
    symptom3 : float
    symptom4 : float
    symptom5 : float
    symptom6 : float
    symptom7 : float
    symptom8 : float
    symptom9 : float
    symptom10 : float
    symptom11 : float
    symptom12 : float
    symptom13 : float
    symptom14 : float
    symptom15 : float
    symptom16 : float
    symptom17 : float


with open("rfc_model.pkl", "rb") as f:
    model = pickle.load(f)


@app.get('/')
def index():
    return {'message': 'This is the homepage of the API '}


@app.post('/prediction')
def get_symptom_category(data: Symptom):
    input = data.json()
    input_dict = json.loads(input)

    symptom1 = input_dict['symptom1']
    symptom2 = input_dict['symptom2']
    symptom3 = input_dict['symptom3']
    symptom4 = input_dict['symptom4']
    symptom5 = input_dict['symptom5']
    symptom6 = input_dict['symptom6']
    symptom7 = input_dict['symptom7']
    symptom8 = input_dict['symptom8']
    symptom9 = input_dict['symptom9']
    symptom10 = input_dict['symptom10']
    symptom11 = input_dict['symptom11']
    symptom12 = input_dict['symptom12']
    symptom13 = input_dict['symptom13']
    symptom14 = input_dict['symptom14']
    symptom15 = input_dict['symptom15']
    symptom16 = input_dict['symptom16']
    symptom17 = input_dict['symptom17']

    input_list = [symptom1, symptom2, symptom3, symptom4, symptom5, symptom6, symptom7, symptom8, symptom9, symptom10, symptom11, symptom12, symptom13, symptom14, symptom15, symptom16, symptom17]

    pred_name = model.predict([input_list])


    return {'prediction': pred_name}


@app.get('/predict')
def get_cat(symptom1 : float, symptom2 : float, symptom3 : float, symptom4 : float, symptom5 : float, symptom6 : float, symptom7 : float, symptom8 : float, symptom9 : float, symptom10 : float, symptom11 : float, symptom12 : float, symptom13 : float, symptom14 : float, symptom15 : float, symptom16 : float, symptom17 : float):
    input_list = [symptom1, symptom2, symptom3, symptom4, symptom5, symptom6, symptom7, symptom8, symptom9, symptom10, symptom11, symptom12, symptom13, symptom14, symptom15, symptom16, symptom17]
    pred_name = model.predict([input_list])
    return {'prediction': pred_name}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)



