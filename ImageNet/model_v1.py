from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Fungsi untuk memuat model TFLite berdasarkan jenis hewan
def load_tflite_model(jenis_hewan):
    # path model berdasarkan jenis hewan
    path_model = f'{jenis_hewan}_ModelMobileNet.tflite'
    
    # Memuat model TFLite
    interpreter = tf.lite.Interpreter(model_path=path_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    return interpreter, input_details, output_details

@app.route('/', methods=['GET'])
def home():
    return 'Service API Aktif'

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Menerima gambar dan jenis hewan dari pengguna
        file_gambar = request.files['image']
        jenis_hewan = request.form.get('type', 'kambing')  # Default ke 'kambing' jika tidak spesifik
        
        # Memuat dan memproses gambar
        gambar = Image.open(file_gambar).convert('RGB')
        gambar = gambar.resize((180, 180))
        data_input = np.expand_dims(np.array(gambar) / 255.0, axis=0).astype(np.float32)

        # Memuat model TFLite berdasarkan jenis hewan
        interpreter, input_details, output_details = load_tflite_model(jenis_hewan)

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
            # Menambahkan label jika ada jenis hewan lain
        }

        label_prediksi = label.get(jenis_hewan, {}).get(kelas_terprediksi, 'Hewan/Kelas Tidak Dikenal')

        hasil = {
            'label_prediksi': label_prediksi,
            'confidence': float(confidence)
        }

        return jsonify(hasil)

    except Exception as e:
        # Menangani exception yang mungkin terjadi selama proses prediksi
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Menjalankan aplikasi Flask pada host='0.0.0.0' dan port=5000
    app.run(host='0.0.0.0', port=5000)
