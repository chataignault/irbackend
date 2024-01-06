from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from source.analytics import *
from source.yf_router import yf_router


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
