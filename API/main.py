from fastapi import FastAPI, File , UploadFile
import numpy as np
from PIL import Image
from io import BytesIO
import uvicorn
import tensorflow as tf
from tensorflow import keras
app = FastAPI()

MODEL = tf.keras.models.load_model(r"C:\Users\91824\Downloads\potatodiseasedetection\models\1.h5")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch=np.expand_dims(image,0)
    predictions=MODEL.predict(img_batch)
    
    index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[index]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app,host='localhost',port=8000)
