# /app/model.py
from config import config
import io
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = config.MODEL_PATH
IMAGE_SIZE = config.IMAGE_SIZE
model = YOLO(MODEL_PATH)
class_names = config.class_names

# skin
def classification(image_bytes: bytes):
    try:
        # 이미지 입력
        image = Image.open(io.BytesIO(image_bytes))
        # 예측
        results = model(image, imgsz = IMAGE_SIZE, verbose=False)
        result = results[0]
        probs = result.probs
        ans_idx = probs.top1
        conf = probs.top1conf.item()
        predicted_class_name = class_names[ans_idx]

        # 모든 클래스별 예측 값
        all_probs = probs.data.tolist()
        all_predict = {
            class_names[i]: prob for i, prob in enumerate(all_probs)
        }

        return {
            "predicted_class": predicted_class_name,
            "confidence": f"{conf:.4f}",
            "class_index": ans_idx,
            "all_predict": all_predict, 
        }

    except Exception as e:
        print(f"Error : {e}")
