import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from io import BytesIO
import os
from PIL import Image

MODEL = tf.keras.models.load_model(r'C:\Users\AUDRIK\OneDrive\Desktop\Plant Disease\plant_models\4')

app = Flask(__name__)

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

CLASS_NAMES = ['Corn Cercospora leaf_spot ',
 'Corn Common rust',
 'Corn Northern Leaf Blight',
 'Corn healthy',
 'Bell Pepper Bacterial_spot',
 'Bell Pepper healthy',
 'Potato Early_blight',
 'Potato Late_blight',
 'Potato healthy',
 'Tomato Bacterial_spot',
 'Tomato Early_blight',
 'Tomato Late_blight',
 'Tomato Leaf_Mold',
 'Tomato Septoria_leaf_spot',
 'Tomato Curl_Virus',
 'Tomato mosaic_virus',
 'Tomato healthy']

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            image = read_file_as_image(file.read())
            img_batch = np.expand_dims(image, 0)
            predictions = MODEL.predict(img_batch)
            predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
            confidence = np.max(predictions[0])
            return render_template('index.html', prediction=predicted_class, confidence=confidence)

    return render_template("index.html", prediction=None, confidence=None)

if __name__ == "__main__":
    app.run()
