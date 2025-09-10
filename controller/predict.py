# /controller/predict.py

from fastapi import APIRouter, UploadFile, File
from models import model

router = APIRouter(
    prefix="/predict", 
    tags=["predict"], 
    responses={404 : {"disciption":"Not Found"}}, 
)

@router.post("")
async def predict(file:UploadFile = File(..., description="분류할 이미지 파일을 업로드 하세요.")):
    image_bytes = await file.read()
    result = model.classification(image_bytes)
    return result
