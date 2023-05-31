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

    psy1 = input_dict['symptom1']
    psy2 = input_dict['symptom2']
    psy3 = input_dict['symptom3']
    psy4 = input_dict['symptom4']
    psy5 = input_dict['symptom5']
    psy6 = input_dict['symptom6']
    psy7 = input_dict['symptom7']
    psy8 = input_dict['symptom8']
    psy9 = input_dict['symptom9']
    psy10 = input_dict['symptom10']
    psy11 = input_dict['symptom11']
    psy12 = input_dict['symptom12']
    psy13 = input_dict['symptom13']
    psy14 = input_dict['symptom14']
    psy15 = input_dict['symptom15']
    psy16 = input_dict['symptom16']
    psy17 = input_dict['symptom17']

    input_list = [psy1, psy2, psy3, psy4, psy5, psy6, psy7, psy8, psy9, psy10, psy11, psy12, psy13, psy14, psy15, psy16, psy17]

    pred_name = model.predict([input_list])


    return {'prediction': pred_name}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)



