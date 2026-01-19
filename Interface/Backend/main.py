from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = tf.keras.models.load_model("ocr_model2.h5")

class_names = [
    'क','ख','ग','घ','ङ','च','छ','ज','झ','ञ',
    'ट','ठ','ड','ढ','ण','त','थ','द','ध','न',
    'प','फ','ब','भ','म','य','र','ल','व',
    'श','ष','स','ह','क्ष','त्र','ज्ञ',
    '०','१','२','३','४','५','६','७','८','९'
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # EXACT SAME loading as your working script
    img = tf.keras.utils.load_img(
        io.BytesIO(contents),
        color_mode="grayscale",
        target_size=(32, 32)
    )

    img = tf.keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)  # (1, 32, 32, 1)

    # DEBUG (keep this for now)
    print("Shape:", img.shape)
    print("Dtype:", img.dtype)
    print("Min:", img.min(), "Max:", img.max())

    preds = model.predict(img)
    scores = tf.nn.softmax(preds, axis=1).numpy()[0]

    idx = int(np.argmax(scores))

    return {
        "character": class_names[idx],
        "confidence": float(scores[idx]) * 100
    }
