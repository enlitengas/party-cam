import numpy as np
import cv2
import time
from flask import Flask, render_template, jsonify, request
import json
import base64
from io import BytesIO
from PIL import Image
import random
from waitress import serve
import os

# Import Hailo SDK
from hailo_platform import HEF
from hailo_platform import ConfigureParams
from hailo_platform import InputVStreamParams, OutputVStreamParams, FormatType
from hailo_platform import VDevice
from hailo_platform import HailoStreamInterface, InferVStreams

# --- Configuration ---
app = Flask(__name__)

# Global variables
modes = ['detection', 'segmentation', 'pose']
sizes = ['s']  # Hailo optimized models typically come in one size
current_mode = random.choice(modes)  # Randomly select initial mode
current_size = 's'
conf_threshold = 0.25
models = {}
fps = 0
detection_counts = {}
hailo_device = None

# Available modes and model paths
model_paths = {
    'detection': 'hailo_models/yolov5s.hef',
    'segmentation': 'hailo_models/yolov5s-seg.hef',
    'pose': 'hailo_models/yolov5s-pose.hef'
}

# COCO class names
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

# --- Hailo Model Class ---
class HailoYOLO:
    def __init__(self, hef_path, mode):
        self.mode = mode
        self.hef_path = hef_path
        self.device = hailo_device
        self.hef = HEF(hef_path)
        self.configure_params = ConfigureParams.create_from_hef(self.hef, device_id=self.device.device_id)
        self.network_group = self.device.create_network_group('yolo_net')
        self.network_group.configure(self.hef, self.configure_params)
        
        # Get input and output tensors info
        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.output_vstream_infos = self.hef.get_output_vstream_infos()
        
        # Input shape
        self.input_height = self.input_vstream_info.shape[2]
        self.input_width = self.input_vstream_info.shape[3]
        self.input_shape = (self.input_width, self.input_height)
        
        # Activate the network
        self.network_group.activate()
        print(f"Model {hef_path} loaded successfully.")
    
    def __call__(self, img):
        # Preprocess image
        preprocessed = self.preprocess(img)
        
        # Run inference
        input_data = {self.input_vstream_info.name: preprocessed}
        outputs = self.network_group.infer(input_data)
        
        # Post-process results based on mode
        if self.mode == 'detection':
            return self.postprocess_detection(outputs, img.shape)
        elif self.mode == 'segmentation':
            return self.postprocess_segmentation(outputs, img.shape)
        elif self.mode == 'pose':
            return self.postprocess_pose(outputs, img.shape)
        
    def preprocess(self, img):
        # Resize image to model input size
        resized = cv2.resize(img, self.input_shape)
        
        # Convert to RGB if needed (Hailo models typically expect RGB)
        if img.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values to 0-1
        normalized = resized.astype(np.float32) / 255.0
        
        # Transpose to NCHW format (batch, channels, height, width)
        transposed = normalized.transpose(2, 0, 1)
        
        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)
        
        return batched
    
    def postprocess_detection(self, outputs, original_shape):
        # Extract detection outputs
        # Note: The exact output tensor names and formats will depend on your specific Hailo model
        detections = []
        
        # This is a simplified example - you'll need to adapt this to your specific model's output format
        detection_output = list(outputs.values())[0]  # Assuming single output tensor
        
        # Process detections (boxes, scores, classes)
        for detection in detection_output:
            if detection[4] < conf_threshold:  # confidence score
                continue
                
            # Extract box coordinates (normalized 0-1)
            x1, y1, x2, y2 = detection[0:4]
            
            # Scale to original image dimensions
            orig_h, orig_w = original_shape[0:2]
            x1 = int(x1 * orig_w)
            y1 = int(y1 * orig_h)
            x2 = int(x2 * orig_w)
            y2 = int(y2 * orig_h)
            
            # Get class with highest confidence
            class_id = int(detection[5])
            confidence = float(detection[4])
            
            detections.append({
                'box': [x1, y1, x2, y2],
                'confidence': confidence,
                'class_id': class_id,
                'class_name': class_names[class_id],
                'track_id': None
            })
            
        return detections
    
    def postprocess_segmentation(self, outputs, original_shape):
        # This is a simplified example - adapt to your specific model's output format
        detections = self.postprocess_detection(outputs, original_shape)
        
        # Extract mask data if available
        mask_output = outputs.get('masks', None)  # Adjust tensor name as needed
        
        if mask_output is not None:
            orig_h, orig_w = original_shape[0:2]
            
            for i, detection in enumerate(detections):
                # Extract and process mask for this detection
                mask = mask_output[i]  # Simplified - adjust based on your model's output format
                
                # Resize mask to original image dimensions
                mask = cv2.resize(mask, (orig_w, orig_h))
                
                # Convert to polygon points for frontend rendering
                # This is a simplified approach - you may need more sophisticated contour extraction
                contours, _ = cv2.findContours((mask > 0.5).astype(np.uint8), 
                                              cv2.RETR_EXTERNAL, 
                                              cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Get the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # Simplify the contour to reduce data size
                    epsilon = 0.005 * cv2.arcLength(largest_contour, True)
                    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                    
                    # Convert to list of [x,y] points
                    polygon = [[float(p[0][0]), float(p[0][1])] for p in approx]
                    
                    # Add to detection
                    detection['mask'] = polygon
        
        return detections
    
    def postprocess_pose(self, outputs, original_shape):
        # This is a simplified example - adapt to your specific model's output format
        detections = self.postprocess_detection(outputs, original_shape)
        
        # Extract keypoints data if available
        keypoints_output = outputs.get('keypoints', None)  # Adjust tensor name as needed
        
        if keypoints_output is not None:
            orig_h, orig_w = original_shape[0:2]
            
            for i, detection in enumerate(detections):
                # Extract keypoints for this detection
                keypoints = keypoints_output[i]  # Simplified - adjust based on your model's output format
                
                # Process keypoints (typically x, y, confidence for each keypoint)
                processed_keypoints = []
                processed_keypoints_conf = []
                
                for kp in keypoints:
                    # Scale to original image dimensions
                    x = float(kp[0] * orig_w)
                    y = float(kp[1] * orig_h)
                    conf = float(kp[2])
                    
                    processed_keypoints.append([x, y])
                    processed_keypoints_conf.append(conf)
                
                # Add to detection
                detection['keypoints'] = processed_keypoints
                detection['keypoints_conf'] = processed_keypoints_conf
        
        return detections

# --- Model Loading ---
def initialize_hailo():
    """Initialize Hailo device"""
    global hailo_device
    try:
        # Create Hailo device context
        hailo_device = VDevice()
        print("Hailo device initialized successfully.")
        return True
    except Exception as e:
        print(f"Error initializing Hailo device: {e}")
        return False

def load_model(mode):
    """Loads a Hailo model based on mode"""
    if mode not in model_paths:
        print(f"Error: Unknown mode {mode}")
        return False
    
    model_path = model_paths[mode]
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        return False
    
    print(f"Loading model: {model_path} for mode: {mode}")
    try:
        models[mode] = HailoYOLO(model_path, mode)
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def load_models():
    """Load models for all modes"""
    # Initialize Hailo device first
    if not initialize_hailo():
        print("Failed to initialize Hailo device. Exiting.")
        exit(1)
    
    print(f"Loading initial models for size: {current_size}...")
    for mode in modes:
        load_model(mode)

# --- Frame Processing ---
def process_frame(frame):
    """Process a frame with current model and return results"""
    global fps
    
    start_time = time.time()
    
    # Ensure the current mode model is loaded
    if current_mode not in models:
        print(f"Model for {current_mode} not loaded. Loading now...")
        if not load_model(current_mode):
            return []
    
    # Process frame with model
    results = models[current_mode](frame)
    
    # Update FPS calculation
    process_time = time.time() - start_time
    fps = 1.0 / process_time if process_time > 0 else 0
    
    # Update detection counts
    detection_counts.clear()
    for result in results:
        class_name = result['class_name']
        if class_name in detection_counts:
            detection_counts[class_name] += 1
        else:
            detection_counts[class_name] = 1
    
    return results

# --- Flask Routes ---
@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html', 
                          modes=modes,
                          current_mode=current_mode,
                          sizes=sizes,
                          current_size=current_size,
                          conf_threshold=conf_threshold)

@app.route('/process_image', methods=['POST'])
def process_image_route():
    """Process an image from the client"""
    try:
        # Get image data from request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data received'}), 400
        
        # Decode base64 image
        image_data = data['image']
        # Remove the data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 to image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to OpenCV format
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Process the frame
        results = process_frame(frame)
        
        # Return results as JSON
        return jsonify(results)
    
    except Exception as e:
        print(f"Error in /process_image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_stats', methods=['GET'])
def get_stats():
    """Return current stats"""
    return jsonify({
        'fps': round(fps, 1),
        'mode': current_mode,
        'size': current_size,
        'conf': conf_threshold,
        'detections': detection_counts
    })

@app.route('/set_mode', methods=['POST'])
def set_mode():
    """Set the processing mode"""
    global current_mode
    
    try:
        data = request.get_json()
        if not data or 'mode' not in data:
            return jsonify({'error': 'No mode specified'}), 400
        
        new_mode = data['mode']
        if new_mode not in modes:
            return jsonify({'error': f'Invalid mode: {new_mode}'}), 400
        
        # Set the new mode
        current_mode = new_mode
        print(f"Mode changed to: {current_mode}")
        
        # Ensure model is loaded
        if current_mode not in models:
            load_model(current_mode)
        
        return jsonify({'success': True, 'mode': current_mode})
    
    except Exception as e:
        print(f"Error in /set_mode: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/set_conf', methods=['POST'])
def set_conf():
    """Set the confidence threshold"""
    global conf_threshold
    
    try:
        data = request.get_json()
        if not data or 'conf' not in data:
            return jsonify({'error': 'No confidence threshold specified'}), 400
        
        new_conf = float(data['conf'])
        if new_conf < 0 or new_conf > 1:
            return jsonify({'error': 'Confidence threshold must be between 0 and 1'}), 400
        
        # Set the new confidence threshold
        conf_threshold = new_conf
        print(f"Confidence threshold changed to: {conf_threshold}")
        
        return jsonify({'success': True, 'conf': conf_threshold})
    
    except Exception as e:
        print(f"Error in /set_conf: {str(e)}")
        return jsonify({'error': str(e)}), 500

# --- Application Startup ---
if __name__ == '__main__':
    # Create hailo_models directory if it doesn't exist
    os.makedirs('hailo_models', exist_ok=True)
    
    # Load models
    load_models()
    
    print("Starting server...")
    # Use waitress for production-grade serving
    serve(app, host='0.0.0.0', port=5100)
