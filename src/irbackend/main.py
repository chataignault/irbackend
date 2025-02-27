from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..utils import *
from .routers.yf_router import yf_router
from .routers.ka_router import ka_router

app = FastAPI(title="IR app backend")
app.include_router(yf_router)
app.include_router(ka_router)

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
