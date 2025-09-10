# app.py

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from controller import predict

app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.include_router(predict.router)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    context = {
        "request" : request, 
        "title" : "YOLO v8", 
        "description" : "6-class Classification", 
    }
    return templates.TemplateResponse(
        name="home.html", 
        context=context
    )
