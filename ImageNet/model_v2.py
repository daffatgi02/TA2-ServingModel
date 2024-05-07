from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from io import BytesIO
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = load_model('vgg16_transfer_model.h5')

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected for uploading'}), 400

    try:
        # Convert the FileStorage object to a BytesIO
        img_bytes = BytesIO(file.read())

        # Read the image file and prepare it for prediction
        img = image.load_img(img_bytes, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Model expects images in a batch
        img_array /= 255.0  # Normalize to match training preprocessing

        # Predict
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction, axis=1)[0]

        # Convert class index to label
        labels = {0: 'mata kambing terjangkit pinkeye', 1: 'mata kambing sehat'}
        result = labels[class_idx]
        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
