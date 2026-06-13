import cv2
import numpy as np
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

print("✓ API Ready - Model will load on first prediction")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

@app.route('/info', methods=['GET'])
def info():
    return jsonify({
        "model": "YOLOv8 Parking Detection",
        "framework": "PyTorch"
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Send image file to get parking predictions
    curl -X POST -F "file=@image.jpg" http://localhost:5000/predict
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Invalid image"}), 400
        
        # Placeholder response - your model inference goes here
        return jsonify({
            "success": True,
            "message": "Model loaded. Replace this with your YOLO inference.",
            "image_shape": img.shape
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
