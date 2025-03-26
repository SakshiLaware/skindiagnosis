from flask import render_template, jsonify, Flask, redirect, url_for, request, make_response
import os
import io
import numpy as np
from PIL import Image
import keras.utils as image
from keras.models import model_from_json

app = Flask(__name__)

SKIN_CLASSES = {
    0: {'name': 'Actinic Keratoses', 'symptoms': ['Scaly skin', 'Red patches'], 'severe': False},
    1: {'name': 'Basal Cell Carcinoma', 'symptoms': ['Shiny bump', 'Sore that won’t heal'], 'severe': True},
    2: {'name': 'Benign Keratosis', 'symptoms': ['Thick, scaly growth', 'No pain'], 'severe': False},
    3: {'name': 'Dermatofibroma', 'symptoms': ['Firm, red or brown nodule', 'No pain'], 'severe': False},
    4: {'name': 'Melanoma', 'symptoms': ['Irregular dark spots', 'Bleeding or itching'], 'severe': True},
    5: {'name': 'Melanocytic Nevi', 'symptoms': ['Flat or raised mole', 'Symmetrical shape'], 'severe': False},
    6: {'name': 'Vascular skin lesion', 'symptoms': ['Red or purple patches', 'Can be painful'], 'severe': False},
    7: {'name': 'Squamous Cell Carcinoma', 'symptoms': ['Red, scaly patches', 'Open sores'], 'severe': True},
    8: {'name': 'Seborrheic Keratosis', 'symptoms': ['Waxy, stuck-on look', 'Varied colors'], 'severe': False},
    9: {'name': 'Lentigo Maligna', 'symptoms': ['Flat, irregular dark spots', 'Grows slowly'], 'severe': True},
    10: {'name': 'Tinea Corporis (Ringworm)', 'symptoms': ['Circular, red rash', 'Itchy skin'], 'severe': False},
    11: {'name': 'Psoriasis', 'symptoms': ['Thick, silvery scales', 'Itching or burning'], 'severe': False},
    12: {'name': 'Eczema', 'symptoms': ['Red, dry, itchy skin', 'Inflammation'], 'severe': False},
    13: {'name': 'Lupus Rash', 'symptoms': ['Butterfly-shaped rash', 'Painful swelling'], 'severe': True},
    14: {'name': 'Rosacea', 'symptoms': ['Facial redness', 'Visible blood vessels'], 'severe': False},
    15: {'name': 'Kaposi Sarcoma', 'symptoms': ['Purple or red lesions', 'Swelling'], 'severe': True},
    16: {'name': 'Impetigo', 'symptoms': ['Red sores', 'Yellow crust'], 'severe': False},
    17: {'name': 'Cutaneous T-cell Lymphoma', 'symptoms': ['Scaly patches', 'Swollen lymph nodes'], 'severe': True},
    18: {'name': 'Neurofibromatosis', 'symptoms': ['Soft bumps', 'Pigmented spots'], 'severe': True},
    19: {'name': 'Alopecia Areata', 'symptoms': ['Sudden hair loss', 'Small bald patches'], 'severe': False}
}

def findMedicine(pred):
    medicines = {
        0: "Fluorouracil",
        1: "Aldara",
        2: "Hydrogen Peroxide",
        3: "Fluorouracil",
        4: "Fluorouracil (5-FU)",
        5: "Fluorouracil",
        6: "Fluorouracil",
        7: "Imiquimod",
        8: "Salicylic Acid",
        9: "Topical Retinoids",
        10: "Antifungal Cream",
        11: "Steroid Cream",
        12: "Moisturizers & Steroids",
        13: "Hydroxychloroquine",
        14: "Metronidazole Gel",
        15: "Chemotherapy",
        16: "Antibiotics",
        17: "Radiation Therapy",
        18: "Genetic Counseling",
        19: "Corticosteroids"
    }
    return medicines.get(pred, "Consult a dermatologist")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signin')
def signin():
    return render_template('signin.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    return render_template('dashboard.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    json_response = {}
    if request.method == 'POST':
        try:
            file = request.files['file']
        except KeyError:
            return make_response(jsonify({'error': 'No file part in the request', 'code': 'FILE', 'message': 'file is not valid'}), 400)

        imagePil = Image.open(io.BytesIO(file.read()))
        imageBytesIO = io.BytesIO()
        imagePil.save(imageBytesIO, format='JPEG')
        imageBytesIO.seek(0)

        j_file = open('model.json', 'r')
        loaded_json_model = j_file.read()
        j_file.close()
        model = model_from_json(loaded_json_model)
        model.load_weights('model.h5')

        img = image.load_img(imageBytesIO, target_size=(224, 224))
        img = np.array(img).reshape((1, 224, 224, 3)) / 255

        prediction = model.predict(img)
        pred = np.argmax(prediction)
        disease_info = SKIN_CLASSES.get(pred, {'name': 'Unknown', 'symptoms': [], 'severe': False})
        accuracy = round(prediction[0][pred] * 100, 2)
        medicine = findMedicine(pred)

        json_response = {
            "detected": True,
            "disease": disease_info['name'],
            "accuracy": accuracy,
            "symptoms": disease_info['symptoms'],
            "severe": disease_info['severe'],
            "medicine": medicine,
            "img_path": file.filename,
        }

        return make_response(jsonify(json_response), 200)
    
    return render_template('detect.html')

if __name__ == "__main__":
    app.run(debug=True, port=3000)