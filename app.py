import numpy as np
import cv2
import time
from flask import Flask, render_template, jsonify, request
from ultralytics import YOLO
import json
import base64
from io import BytesIO
from PIL import Image
import random

# --- Configuration ---
app = Flask(__name__)

# Global variables
modes = ['detection', 'segmentation', 'pose']
sizes = ['n', 's', 'm', 'l', 'x']
current_size = 'x'
conf_threshold = 0.25
models = {}
fps = 0
detection_counts = {}

# Available modes and model sizes
class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]
colors = np.random.uniform(0, 255, size=(len(class_names), 3))

# --- Model Loading ---
def load_model(mode, size):
    """Loads a YOLO model based on mode and size"""
    model_name = f'yolo11{size}'
    if mode == 'segmentation':
        model_name += '-seg'
    elif mode == 'pose':
        model_name += '-pose'
    model_name += '.pt'

    print(f"Loading model: {model_name} for mode: {mode}")
    try:
        models[mode] = YOLO(model_name)
        print(f"Model {model_name} loaded successfully.")
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")

def load_models():
    """Loads models for the current mode and potentially others if needed"""
    print(f"Loading initial models for size: {current_size}...")
    for mode in modes:
        try:
            load_model(mode, current_size)
        except Exception as e:
            print(f"Error loading model for mode {mode} size {current_size}: {e}")

# --- Frame Processing ---
def process_frame(frame):
    """
    Process a frame with current model and settings.
    Returns a list of detection/tracking results, not the annotated frame.
    """
    results_list = []
    current_detection_counts = {}

    model_to_use = models.get(current_mode)
    if not model_to_use:
        print(f"Error: Model for mode '{current_mode}' not loaded.")
        return []

    try:
        if current_mode == "detection":
            results = model_to_use(frame, conf=conf_threshold, verbose=False)
            result = results[0]

            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = class_names[class_id]

                current_detection_counts[class_name] = current_detection_counts.get(class_name, 0) + 1

                results_list.append({
                    'box': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                })

        elif current_mode == "tracking":
            results = model_to_use.track(frame, persist=True, conf=conf_threshold, verbose=False)
            result = results[0] if results else None

            if result and hasattr(result, 'boxes') and hasattr(result.boxes, 'id') and result.boxes.id is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = class_names[class_id]
                    track_id = int(box.id[0])

                    current_detection_counts[class_name] = current_detection_counts.get(class_name, 0) + 1

                    results_list.append({
                        'box': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name,
                        'track_id': track_id
                    })
            elif result:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = class_names[class_id]
                    current_detection_counts[class_name] = current_detection_counts.get(class_name, 0) + 1
                    results_list.append({
                        'box': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name,
                        'track_id': None
                    })

        elif current_mode == "segmentation":
            results = model_to_use(frame, conf=conf_threshold, verbose=False)
            result = results[0]

            if result.masks is not None:
                for i, box in enumerate(result.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = class_names[class_id]
                    mask_points = result.masks.xy[i].astype(int).tolist()

                    current_detection_counts[class_name] = current_detection_counts.get(class_name, 0) + 1

                    results_list.append({
                        'box': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name,
                        'mask': mask_points
                    })

        elif current_mode == "pose":
            results = model_to_use(frame, conf=conf_threshold, verbose=False)
            result = results[0]

            if result.keypoints is not None:
                for i, box in enumerate(result.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = class_names[class_id]
                    keypoints = result.keypoints.xy[i].cpu().numpy().astype(int).tolist()
                    keypoints_conf = result.keypoints.conf[i].cpu().numpy().tolist() if result.keypoints.conf is not None else None

                    current_detection_counts[class_name] = current_detection_counts.get(class_name, 0) + 1

                    results_list.append({
                        'box': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name,
                        'keypoints': keypoints,
                        'keypoints_conf': keypoints_conf
                    })
    except Exception as e:
        print(f"Error processing frame in mode '{current_mode}': {e}")
        return []

    return results_list, current_detection_counts

# --- Flask Routes ---
@app.route('/')
def index():
    """Render the main page"""
    global current_mode
    current_mode = random.choice(modes)
    load_model(current_mode, current_size)
    print(f"Initial mode selected randomly: {current_mode}")
    return render_template('index.html', modes=modes, sizes=sizes,
                           current_mode=current_mode, current_size=current_size,
                           conf_threshold=conf_threshold)

@app.route('/process_image', methods=['POST'])
def process_image_route():
    """Receives image data, processes it, and returns results"""
    global detection_counts

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'Missing image data'}), 400

    try:
        image_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(image_data)
        pil_image = Image.open(BytesIO(img_bytes))
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        original_height, original_width = frame.shape[:2]

        start_time = time.time()
        results_list, frame_counts = process_frame(frame)
        end_time = time.time()

        processing_time = (end_time - start_time) * 1000

        detection_counts = frame_counts

        return jsonify({
            'results': results_list,
            'processing_time_ms': processing_time,
            'original_width': original_width,
            'original_height': original_height
        })

    except base64.binascii.Error as e:
        return jsonify({'error': f'Invalid base64 data: {e}'}), 400
    except Exception as e:
        print(f"Error in /process_image: {e}")
        return jsonify({'error': f'Error processing image: {e}'}), 500

@app.route('/get_stats')
def get_stats():
    """Return current detection statistics"""
    return jsonify({'detections': detection_counts})

@app.route('/set_mode', methods=['POST'])
def set_mode():
    """Set the processing mode"""
    global current_mode, current_size
    data = request.get_json()
    new_mode = data.get('mode')
    new_size = data.get('size')

    if new_mode in modes:
        current_mode = new_mode
        print(f"Mode changed to: {current_mode}")
        if current_mode not in models or data.get('force_reload'):
            load_model(current_mode, new_size if new_size else current_size)
        return jsonify({'status': 'success', 'mode': current_mode})
    else:
        return jsonify({'status': 'error', 'message': 'Invalid mode'}), 400

@app.route('/set_conf', methods=['POST'])
def set_conf():
    """Set the confidence threshold"""
    global conf_threshold
    data = request.get_json()
    try:
        new_conf = float(data.get('conf'))
        if 0.0 <= new_conf <= 1.0:
            conf_threshold = new_conf
            print(f"Confidence threshold set to: {conf_threshold}")
            return jsonify({'status': 'success', 'conf': conf_threshold})
        else:
            return jsonify({'status': 'error', 'message': 'Confidence must be between 0.0 and 1.0'}), 400
    except (ValueError, TypeError):
        return jsonify({'status': 'error', 'message': 'Invalid confidence value'}), 400

# --- Application Startup ---
load_models()
if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5100, debug=False, threaded=True, use_reloader=False)
