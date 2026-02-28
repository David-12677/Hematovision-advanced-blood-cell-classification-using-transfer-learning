import os
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# LOAD YOUR MODEL: Make sure 'blood_cell_model.h5' is in the 'model' folder
# If you don't have a model yet, this line will cause an error.
try:
    model = tf.keras.models.load_model('blood_cell_model.h5')
except:
    model = None
    print("Warning: Model file not found in 'blood_cell_model.h5'")

# These are the 4 main types of white blood cells
classes = ['Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file selected"
    
    file = request.files['file']
    if file.filename == '':
        return "No file selected"

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    if model is None:
        return "Error: AI Model not loaded. Please train the model first."

    # Preparing the image for the AI (224x224 is standard for Transfer Learning)
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # AI making the prediction
    preds = model.predict(img_array)
    best_index = np.argmax(preds[0])
    label = classes[best_index]
    confidence = round(100 * np.max(preds[0]), 2)

    return render_template('result.html', label=label, confidence=confidence, img_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)