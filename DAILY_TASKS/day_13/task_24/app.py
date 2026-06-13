import base64
import os
import tempfile
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, render_template_string, request
from ultralytics import YOLO

app = Flask(__name__)

# Resolve the path to the best YOLO weights from the Video Pipeline
WEIGHTS_PATH = (
    Path(__file__).resolve().parents[3]
    / "AI_Based_Parking_Management_System"
    / "Video"
    / "runs"
    / "parking_detector"
    / "parksight_yolov8s"
    / "weights"
    / "best.pt"
)

# Load the model globally so it's loaded only once when the server starts
try:
    model = YOLO(str(WEIGHTS_PATH))
except Exception as e:
    model = None
    print(f"Warning: Could not load YOLO model from {WEIGHTS_PATH}. Error: {e}")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ParkSight Intelligence</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
        body { font-family: 'Inter', sans-serif; background-color: #eef3f6; }
        /* Fixes the "so big" annotation issue by strictly capping the max height */
        .media-container img { max-height: 65vh; width: auto; margin: 0 auto; border-radius: 0.75rem; object-fit: contain; }
        .loader { border-top-color: #0f766e; -webkit-animation: spinner 1.5s linear infinite; animation: spinner 1.5s linear infinite; }
        @keyframes spinner { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
    <script>
        function updateFileName(input) {
            const fileNameSpan = document.getElementById('file-name');
            if (input.files && input.files[0]) {
                fileNameSpan.textContent = input.files[0].name;
                fileNameSpan.classList.remove('text-slate-500');
                fileNameSpan.classList.add('text-teal-700', 'font-bold');
            }
        }
        function showLoader() {
            document.getElementById('submit-btn').classList.add('hidden');
            document.getElementById('loading-btn').classList.remove('hidden');
        }
    </script>
</head>
<body class="min-h-screen text-slate-800 flex flex-col items-center py-10 px-4">
    <div class="w-full max-w-5xl bg-white shadow-2xl rounded-3xl overflow-hidden border border-slate-200">
        <!-- Header -->
        <div class="bg-gradient-to-r from-teal-900 to-teal-700 p-8 text-white text-center">
            <h1 class="text-4xl font-black tracking-tight">ParkSight Intelligence</h1>
            <p class="mt-2 text-teal-100 font-medium text-lg">Real-time Parking Occupancy Computer Vision</p>
        </div>
        
        <div class="p-8">
            <!-- Upload Form -->
            <form action="/predict" method="post" enctype="multipart/form-data" class="flex flex-col items-center space-y-6" onsubmit="showLoader()">
                <label class="w-full max-w-2xl flex flex-col items-center px-4 py-10 bg-slate-50 text-slate-500 rounded-2xl border-2 border-dashed border-slate-300 cursor-pointer hover:bg-slate-100 hover:border-teal-500 transition-all duration-200">
                    <svg class="w-12 h-12 mb-4 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path></svg>
                    <span id="file-name" class="text-lg font-medium text-slate-600 text-center break-all px-4">Click to upload an image or video</span>
                    <span class="text-sm mt-2 text-slate-400 font-medium tracking-wide">Supports JPG, PNG, MP4, AVI, MOV</span>
                    <input type="file" name="file" class="hidden" accept="image/*, video/*" required onchange="updateFileName(this)">
                </label>

                <button id="submit-btn" type="submit" class="px-10 py-4 bg-teal-600 hover:bg-teal-700 text-white text-lg font-bold rounded-xl shadow-lg transition transform hover:scale-105 flex items-center gap-2">
                    <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                    Analyze Media
                </button>
                
                <button id="loading-btn" type="button" disabled class="hidden px-10 py-4 bg-teal-500 text-white text-lg font-bold rounded-xl shadow-lg flex items-center gap-3 cursor-not-allowed">
                    <div class="loader ease-linear rounded-full border-4 border-t-4 border-teal-200 h-6 w-6"></div>
                    Processing...
                </button>
            </form>

            <!-- Error Message -->
            {% if error %}
            <div class="mt-8 max-w-2xl mx-auto p-4 bg-red-50 border-l-4 border-red-500 text-red-700 rounded-r-lg shadow-sm">
                <p class="font-bold">System Error</p>
                <p>{{ error }}</p>
            </div>
            {% endif %}

            <!-- Static Image Result -->
            {% if media_type == 'image' %}
            <div class="mt-12">
                <div class="flex items-center justify-center gap-4 mb-8">
                    <div class="h-px bg-slate-200 flex-1"></div>
                    <h2 class="text-2xl font-extrabold text-slate-800">Analysis Result</h2>
                    <div class="h-px bg-slate-200 flex-1"></div>
                </div>
                
                <div class="grid grid-cols-2 gap-6 mb-8 max-w-xl mx-auto text-center">
                    <div class="bg-red-50 border border-red-100 p-6 rounded-2xl shadow-sm">
                        <p class="text-sm text-red-500 font-extrabold uppercase tracking-widest">Occupied Slots</p>
                        <p class="text-5xl font-black text-red-600 mt-2">{{ occupied_count }}</p>
                    </div>
                    <div class="bg-emerald-50 border border-emerald-100 p-6 rounded-2xl shadow-sm">
                        <p class="text-sm text-emerald-500 font-extrabold uppercase tracking-widest">Empty Slots</p>
                        <p class="text-5xl font-black text-emerald-600 mt-2">{{ empty_count }}</p>
                    </div>
                </div>
                <div class="media-container bg-slate-900 p-3 rounded-2xl shadow-2xl">
                    <img src="data:image/jpeg;base64,{{ result_img }}" alt="Detection Result">
                </div>
            </div>
            {% endif %}

            <!-- Live Video Stream Result -->
            {% if media_type == 'video' %}
            <div class="mt-12 flex flex-col items-center">
                <div class="flex items-center justify-center gap-4 mb-4 w-full">
                    <div class="h-px bg-slate-200 flex-1"></div>
                    <h2 class="text-2xl font-extrabold text-slate-800">Live Video Analysis</h2>
                    <div class="h-px bg-slate-200 flex-1"></div>
                </div>
                <p class="text-slate-500 mb-8 text-center font-medium">Streaming processed frames in real-time with continuous detection.</p>
                
                <div class="media-container bg-slate-900 p-3 rounded-2xl shadow-2xl w-full flex justify-center">
                    <img src="/video_feed?path={{ video_path }}" alt="Live Video Feed">
                </div>
                
                <div class="mt-6">
                    <a href="/" class="text-teal-600 hover:text-teal-800 font-bold flex items-center gap-2">
                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
                        Analyze Another File
                    </a>
                </div>
            </div>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""


def generate_video_frames(video_path):
    """Generator function to read video, run YOLO, draw HUD, and yield JPEG frames."""
    cap = cv2.VideoCapture(video_path)
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            # Run YOLO prediction
            results = model.predict(frame, conf=0.35, verbose=False)[0]
            
            occupied_count = 0
            empty_count = 0
            annotated_frame = frame.copy()
            
            if results.boxes is not None and len(results.boxes) > 0:
                classes = results.boxes.cls.cpu().numpy()
                boxes = results.boxes.xyxy.cpu().numpy()
                
                for box, cls in zip(boxes, classes):
                    x1, y1, x2, y2 = map(int, box)
                    if int(cls) == 1:
                        occupied_count += 1
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    else:
                        empty_count += 1
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw a custom HUD panel on top left similar to realtime_occupancy.py
            cv2.rectangle(annotated_frame, (18, 18), (320, 115), (15, 28, 38), -1)
            cv2.rectangle(annotated_frame, (18, 18), (320, 115), (220, 232, 240), 1)
            cv2.putText(annotated_frame, "ParkSight Live Stream", (34, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Occupied Slots: {occupied_count}", (34, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (54, 36, 220), 2)
            cv2.putText(annotated_frame, f"Empty Slots: {empty_count}", (34, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (185, 246, 202), 2)
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            # Yield the frame in multipart format for browser streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()
        # Clean up temporary video file when stream ends or fails
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except:
                pass


@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template_string(HTML_TEMPLATE, error="YOLO model not loaded. Check model path.")

    file = request.files.get('file')
    if not file or file.filename == '':
        return render_template_string(HTML_TEMPLATE, error="No valid file uploaded.")
    
    filename = file.filename.lower()
    is_video = filename.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))
    
    if is_video:
        try:
            # Save video to a temporary directory so OpenCV can read it frame-by-frame
            temp_dir = tempfile.gettempdir()
            safe_name = f"{uuid.uuid4().hex}_{file.filename}"
            save_path = os.path.join(temp_dir, safe_name)
            file.save(save_path)
            
            return render_template_string(HTML_TEMPLATE, media_type='video', video_path=save_path)
        except Exception as e:
            return render_template_string(HTML_TEMPLATE, error=f"Error processing video upload: {str(e)}")
    else:
        try:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                return render_template_string(HTML_TEMPLATE, error="Invalid image file format.")

            # Run YOLO inference on the single image
            results = model.predict(img, conf=0.35, verbose=False)[0]
            
            occupied_count = 0
            empty_count = 0
            annotated_img = img.copy()
            
            if results.boxes is not None and len(results.boxes) > 0:
                classes = results.boxes.cls.cpu().numpy()
                boxes = results.boxes.xyxy.cpu().numpy()
                
                for box, cls in zip(boxes, classes):
                    x1, y1, x2, y2 = map(int, box)
                    if int(cls) == 1:
                        occupied_count += 1
                        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    else:
                        empty_count += 1
                        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            _, buffer = cv2.imencode('.jpg', annotated_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return render_template_string(HTML_TEMPLATE, media_type='image', result_img=img_base64, empty_count=empty_count, occupied_count=occupied_count)
        except Exception as e:
            return render_template_string(HTML_TEMPLATE, error=f"Error processing image: {str(e)}")


@app.route('/video_feed')
def video_feed():
    video_path = request.args.get('path')
    if not video_path or not os.path.exists(video_path):
        return "Video not found", 404
    return Response(generate_video_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=5000)