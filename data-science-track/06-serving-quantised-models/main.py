# app/main.py
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from settings import settings
from api import router, lifespan

# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

app = FastAPI(
  title=settings.app_title,
  version=settings.app_version,
  description=settings.app_description,
  lifespan=lifespan,
  docs_url="/docs",
  redoc_url="/redoc",
)

# CORS
app.add_middleware(
  CORSMiddleware,
  allow_origins=settings.cors_allow_origins,
  allow_methods=settings.cors_allow_methods,
  allow_headers=settings.cors_allow_headers,
  allow_credentials=settings.cors_allow_credentials,
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", include_in_schema=False)
async def home(request: Request):
  return templates.TemplateResponse("index.html", {"request": request})

# include our API router
app.include_router(router)
