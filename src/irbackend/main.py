from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from utils.analytics import *
from utils.yf_router import yf_router


app = FastAPI(title="IR app backend")
app.include_router(yf_router)

origins = [
    "http://localhost:3000/react-tutorial",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
