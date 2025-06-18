from flask import Flask, request, jsonify
import tensorflow as tf
import json
import numpy as np
from PIL import Image
from pathlib import Path

from utils.data_conversion import label_to_json
from config.paths import MODELS_PATH
from config.constants import GAMES 

import os

app = Flask(__name__)

MODEL_PATH = MODELS_PATH / GAMES.LORCANA.value / 'model.keras'
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image, img_width, img_height):
    image = image.resize((img_width, img_height))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        card_index = int(request.form['card_index'])
        img_width = int(request.form['img_width'])
        img_height = int(request.form['img_height'])
        threshold = float(request.form.get('threshold', 0.6))

        image = Image.open(file.stream)
        processed_image = preprocess_image(image, img_width, img_height)

        pred = model.predict(processed_image)
        result_index = np.argmax(pred)
        confidence = pred[0][result_index]

        data = json.loads(label_to_json(pred))
        data.append({'confidence': float(confidence)})
        
        return json.dumps(data)
        
        # do not worry about this for now
        # if confidence < threshold:
        #     return jsonify({
        #         'message': 'Model predicted the image as unknown.',
        #         'confidence': float(confidence)
        #         })
        # else:
        #     message = f'Model predicted index {result_index} with {confidence:.4%} confidence.'
        #     if result_index != card_index:
        #         message += ' (INCORRECT)'

        #     return jsonify({
        #         'message': message,
        #         'confidence': float(confidence)
        #         })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def start_server():
    app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

