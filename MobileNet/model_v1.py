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
import atexit

app = Flask(__name__)
CORS(app)

# Inisialisasi interpreter di luar fungsi load_tflite_model
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

# Fungsi untuk memuat model TFLite berdasarkan jenis hewan jika interpreter belum ada
def load_tflite_model(jenis_hewan):
    global interpreter
    if interpreter is None:
        # path model berdasarkan jenis hewan
        path_model = f'{jenis_hewan}_ModelMobileNet.tflite'
        
        # Memuat model TFLite
        interpreter = tf.lite.Interpreter(model_path=path_model)
        interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    return input_details, output_details

# Memuat model saat aplikasi dimulai
load_tflite_model('kambing')

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
        gambar = gambar.resize((180, 180))
        data_input = np.expand_dims(np.array(gambar) / 255.0, axis=0).astype(np.float32)

        # Memuat model TFLite berdasarkan jenis hewan
        input_details, output_details = load_tflite_model(jenis_hewan)

        # Handling Error 500: Model Load Failed
        if not input_details or not output_details:
            raise Exception('Error 500: Model Load Failed')

        # Menetapkan nilai tensor input
        interpreter.set_tensor(input_details[0]['index'], data_input)

        # Menjalankan proses prediksi
        interpreter.invoke()

        # Mendapatkan nilai tensor output
        hasil_prediksi = interpreter.get_tensor(output_details[0]['index'])
        kelas_terprediksi = np.argmax(hasil_prediksi)
        confidence = hasil_prediksi[0][kelas_terprediksi]

        # Memetakan label sesuai jenis hewan
        label = {
            'kambing': {0: 'Mata Kambing Terlihat Sehat', 1: 'Mata Kambing Terjangkit PinkEye'},
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
