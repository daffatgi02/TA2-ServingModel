from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from flask_cors import CORS
from werkzeug.exceptions import BadRequest
import logging
from datetime import datetime
from pytz import timezone

app = Flask(__name__)
CORS(app)

# Inisialisasi interpreter di luar fungsi load_model_imagenet
interpreter = None

# Inisialisasi logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('backend-ml-ternakami-app')

# Fungsi untuk mendapatkan waktu sekarang dengan timezone Indonesia
def get_current_time():
    time_format = "%Y-%m-%d-%H:%M:%S"
    tz = timezone('Asia/Jakarta')
    current_time = datetime.now(tz)
    return current_time.strftime(time_format)

# Fungsi untuk memuat model ImageNet berdasarkan jenis hewan jika interpreter belum ada
def load_model_imagenet(jenis_hewan):
    global interpreter
    if interpreter is None:
        # Path model berdasarkan jenis hewan
        model_filename = f'{jenis_hewan}_ModelImageNet.h5'
        interpreter = load_model(model_filename)
    
    return interpreter

@app.route('/', methods=['GET'])
def home():
    logger.info(f"{get_current_time()}-LOG STATUS: Service API Aktif")
    return 'Service API Aktif'

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        logger.info(f"{get_current_time()}-LOG STATUS: Received prediction request")
        # Menerima gambar dan jenis hewan dari pengguna
        file_gambar = request.files.get('image')
        jenis_hewan = request.form.get('type')  # jenis hewan "kambing" atau "sapi" tergantung inputan dari client

        # Handling Error 400: No Return no Image or Type specified
        if not file_gambar or not jenis_hewan:
            raise BadRequest('Error 400: No Image or Type specified')

        # Memuat dan memproses gambar
        gambar = Image.open(file_gambar).convert('RGB')
        gambar = gambar.resize((224, 224))
        data_input = np.expand_dims(np.array(gambar) / 255.0, axis=0).astype(np.float32)

        # Memuat model ImageNet berdasarkan jenis hewan
        model = load_model_imagenet(jenis_hewan)

        # Menjalankan proses prediksi
        hasil_prediksi = model.predict(data_input)
        kelas_terprediksi = np.argmax(hasil_prediksi)
        confidence = hasil_prediksi[0][kelas_terprediksi]

        # Memetakan label sesuai jenis hewan
        label = {
            'kambing': {0: 'Mata Terjangkit PinkEye', 1: 'Mata Terlihat Sehat'},
        }

        label_prediksi = label.get(jenis_hewan, {}).get(kelas_terprediksi, 'Hewan/Kelas Tidak Dikenal')

        hasil = {
            'label_prediksi': label_prediksi,
            'confidence': float(confidence)
        }

        logger.info(f"{get_current_time()}-LOG STATUS: Prediction successful")
        return jsonify(hasil)

    except Exception as e:
        # Handling Error 500: Prediction Failed
        logger.error(f"{get_current_time()}-LOG STATUS: Prediction failed - {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Menjalankan aplikasi Flask pada host='0.0.0.0' dan port=5000
    app.run(host='0.0.0.0', port=5000)
