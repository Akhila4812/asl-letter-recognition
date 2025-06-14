from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import tensorflow as tf
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
model = None

def load_model():
    global model
    if model is None:
        print("Loading model...")
        model = tf.keras.models.load_model('asl_model.h5')
        print("Model loaded!")
    return model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        img = Image.open(filepath).convert('L').resize((28, 28))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        model = load_model()
        prediction = model.predict(img_array)
        label = chr(np.argmax(prediction) + 65)  # 0-25 to A-Z
        confidence = float(np.max(prediction) * 100)
        image_path = f"{UPLOAD_FOLDER}/{filename}"
        return render_template('index.html', label=label, confidence=f"{confidence:.2f}", image_path=image_path)
    return jsonify({'error': 'Invalid file'}), 400

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'Flask is working!'})

if __name__ == '__main__':
    app.run(debug=False)