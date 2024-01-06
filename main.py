from fastapi import FastAPI
import json
import yaml
from json import JSONEncoder

import os
import pandas as pd
import numpy as np

from source.analytics import *
from source.data_getter import yf_router

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.include_router(yf_router)

origins = [
    "*",
    "http://localhost:3000/*",
    "http://127.0.0.1:8000/ir_analysis",
    "http://127.0.0.1:8000/ir_data/",
    "http://127.0.0.1:8000/" "http://localhost:3000/react-tutorial",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/data/{file_name}")
def read_csv(file_name: str):
    data = pd.read_csv(os.path.join(os.getcwd(), file_name)).set_index("Prenom")
    print(data)
    print(data.to_dict())
    return data.to_dict()


@app.get("/ir_data/{file_name}")
def read_ir_data_cropped(file_name: str, n_rows: int = 10):
    data = pd.DataFrame({"Open": [1.0, 3.0, 4.0], "Date": [1, 2, 3]})
    return data.iloc[:n_rows].to_json(orient="records")


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


@app.get("/ir_analysis/")
def process_PCA():
    rates = generate_data()
    _, D, V = get_principal_components(rates.values)
    return json.dumps({"array": V}, cls=NumpyArrayEncoder)
