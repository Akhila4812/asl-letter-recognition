from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os
app = Flask(__name__)
model = None
def load_model():
    global model
    if model is None:
        print("Loading model...")
        model = tf.keras.models.load_model('asl_model.h5')
        print("Model loaded!")
    return model
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    img = Image.open(filepath).convert('L').resize((28, 28))
    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28, 1)
    model = load_model()
    pred = model.predict(img)
    confidence = float(np.max(pred))
    label_idx = np.argmax(pred)
    label = chr(label_idx + 65)
    return jsonify({
        'label': label,
        'confidence': f'{confidence:.2%}'
    })
if __name__ == '__main__':
    app.run(debug=False, port=5000)  # debug=False for speed