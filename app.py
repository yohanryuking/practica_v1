from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import cv2
import json
from skimage.measure import regionprops, label

app = Flask(__name__)

# Cargar el modelo y otros recursos necesarios
model = joblib.load('VotingClassifier_best_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

def binarize_image(image, threshold=127):
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

def extract_features(binary_image, num_features=11):
    labeled_image = label(binary_image)
    regions = regionprops(labeled_image)
    
    features = []
    for region in regions:
        perimeter = region.perimeter
        area = region.area
        eccentricity = region.eccentricity
        equivalent_diameter = region.equivalent_diameter
        major_axis_length = region.major_axis_length
        minor_axis_length = region.minor_axis_length
        solidity = region.solidity
        extent = region.extent
        orientation = region.orientation
        
        if perimeter == 0:
            csf = 0
        else:
            csf = (4 * np.pi * area) / (perimeter ** 2)
        
        if minor_axis_length == 0:
            esf = 0
        else:
            esf = major_axis_length / minor_axis_length
        
        features.append([
            perimeter, area, eccentricity, equivalent_diameter,
            major_axis_length, minor_axis_length, solidity, extent,
            orientation, csf, esf
        ])
    
    if len(features) > 0:
        features = np.array(features).flatten()
        if len(features) < num_features:
            features = np.pad(features, (0, num_features - len(features)), 'constant')
        elif len(features) > num_features:
            features = features[:num_features]
    else:
        features = np.zeros(num_features)
    
    return features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400
    
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = binarize_image(img)
    features = extract_features(img)
    features = scaler.transform([features])
    
    prediction = model.predict(features)
    label = label_encoder.inverse_transform(prediction)[0]
    
    return jsonify({'label': label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
