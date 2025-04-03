import cv2
import numpy as np
import time
from ultralytics import YOLO

def main():
    # Define the window dimensions
    window_width = 1280
    window_height = 720
    
    # List of available model sizes (from smallest/fastest to largest/most accurate)
    model_sizes = ['n', 's', 'm', 'l', 'x']
    current_size = 'n'  # Start with nano model
    
    # Available modes
    modes = ['detection', 'segmentation', 'pose', 'tracking']
    current_mode = 'detection'  # Start with detection mode
    
    # Load all the models (lazy loading - will download if not present)
    models = {}
    for mode in modes:
        if mode == 'pose':
            models[mode] = YOLO(f"yolo12{current_size}-pose.pt")
        elif mode == 'segmentation':
            models[mode] = YOLO(f"yolo12{current_size}-seg.pt")
        elif mode in ['detection', 'tracking']:
            models[mode] = YOLO(f"yolo12{current_size}.pt")
    
    # Define class names for COCO dataset (80 classes)
    class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                  'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                  'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                  'hair drier', 'toothbrush']
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
        
    # Set the capture resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, window_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, window_height)
        
    # Set confidence threshold for detections
    conf_threshold = 0.25
    
    # Initialize FPS calculation variables
    prev_time = 0
    curr_time = 0
    fps = 0
    
    # Set up instruction text
    print("=== YOLO Webcam Multi-Mode Application ===")
    print("Press 'q' to quit")
    print("Press 'm' to cycle through modes: detection → segmentation → pose → tracking")
    print("Press 's' to change model size: n → s → m → l → x (affects accuracy and speed)")
    print("Press '+'/'-' to adjust confidence threshold")
    
    # For visualization: generate random colors for each class
    np.random.seed(42)  # for reproducibility
    colors = {}
    for i in range(len(class_names)):
        # Generate random bright colors (BGR format)
        color = (int(np.random.randint(100, 255)), 
                int(np.random.randint(100, 255)), 
                int(np.random.randint(100, 255)))
        colors[i] = color
        
    # Skeleton for pose estimation - pairs of keypoints to connect with lines
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], 
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], 
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    
    # Uncomment if you want to display keypoint names
    # keypoint_names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 
    #                  'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
    #                  'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 
    #                  'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
        
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
     
        # If frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        # Calculate FPS
        curr_time = time.time()
        if prev_time > 0:
            fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        # Check for key presses
        key = cv2.waitKey(1)
        
        # 'q' to quit
        if key == ord('q'):
            break
            
        # 'm' to cycle through modes
        if key == ord('m'):
            # Get the next mode in the cycle
            current_idx = modes.index(current_mode)
            next_idx = (current_idx + 1) % len(modes)
            current_mode = modes[next_idx]
            print(f"Switched to {current_mode} mode")
        
        # 's' to cycle through model sizes
        if key == ord('s'):
            # Get the next model size
            current_idx = model_sizes.index(current_size)
            next_idx = (current_idx + 1) % len(model_sizes)
            current_size = model_sizes[next_idx]
            print(f"Switching to YOLOv8{current_size} models (this may take a moment)")
            
            # Load new models with the new size
            for mode in modes:
                if mode == 'pose':
                    models[mode] = YOLO(f"yolov8{current_size}-pose.pt")
                elif mode == 'segmentation':
                    models[mode] = YOLO(f"yolov8{current_size}-seg.pt")
                elif mode in ['detection', 'tracking']:
                    models[mode] = YOLO(f"yolov8{current_size}.pt")
        
        # '+' or '=' to increase confidence threshold
        if key == ord('+') or key == ord('='):
            conf_threshold = min(conf_threshold + 0.05, 0.95)
            print(f"Confidence threshold: {conf_threshold:.2f}")
            
        # '-' to decrease confidence threshold
        if key == ord('-'):
            conf_threshold = max(conf_threshold - 0.05, 0.05)
            print(f"Confidence threshold: {conf_threshold:.2f}")
        
        # Create a copy of the frame for drawing
        annotated_frame = frame.copy()
        
        # Run inference based on the current mode
        if current_mode == "detection":
            # Object detection mode
            results = models[current_mode](frame, conf=conf_threshold)
            
            # Get the first result
            result = results[0]
            
            # Draw bounding boxes and labels
            for box in result.boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0]  # xmin, ymin, xmax, ymax
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get confidence score
                confidence = float(box.conf[0])
                
                # Get class name
                class_id = int(box.cls[0])
                class_name = class_names[class_id]
                color = colors[class_id]
                
                # Draw rectangle and text
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        elif current_mode == "tracking":
            # Object tracking mode
            results = models[current_mode].track(frame, persist=True, conf=conf_threshold)
            
            # Get the first result
            if results[0] is not None:
                result = results[0]
                
                # Check if we have tracking IDs
                if hasattr(result, 'boxes') and hasattr(result.boxes, 'id') and result.boxes.id is not None:
                    # Draw bounding boxes, labels and tracking IDs
                    for box in result.boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0]  # xmin, ymin, xmax, ymax
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Get confidence score
                        confidence = float(box.conf[0])
                        
                        # Get class name
                        class_id = int(box.cls[0])
                        class_name = class_names[class_id]
                        
                        # Get tracking ID
                        track_id = int(box.id[0])
                        
                        # Generate color based on tracking ID for consistency
                        track_color = (int(hash(str(track_id)) % 256),
                                      int(hash(str(track_id) + '1') % 256),
                                      int(hash(str(track_id) + '2') % 256))
                        
                        # Draw rectangle with track color
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), track_color, 2)
                        
                        # Add label with class, confidence and track ID
                        label = f"{class_name} #{track_id} {confidence:.2f}"
                        cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, track_color, 2)
                
                # If tracking IDs not available, fall back to regular detection display
                else:
                    for box in result.boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Get confidence score and class
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = class_names[class_id]
                        color = colors[class_id]
                        
                        # Draw rectangle and text
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{class_name} {confidence:.2f}"
                        cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        elif current_mode == "segmentation":
            # Segmentation mode
            results = models[current_mode](frame, conf=conf_threshold)
            
            # Get the first result
            result = results[0]
            
            # Create a transparent overlay for the segmentation masks
            overlay = annotated_frame.copy()
            
            # Process segmentation masks and bounding boxes
            if hasattr(result, 'masks') and result.masks is not None:
                for i, mask in enumerate(result.masks):
                    # Get corresponding box and class information
                    box = result.boxes[i]
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = class_names[class_id]
                    color = colors[class_id]
                    
                    # Get segmentation mask as numpy array
                    segment = mask.data[0].cpu().numpy()  # Get the mask data
                    segment = cv2.resize(segment, (frame.shape[1], frame.shape[0]))
                    
                    # Create binary mask from the segment
                    contours = cv2.findContours((segment > 0.5).astype(np.uint8), 
                                              cv2.RETR_EXTERNAL, 
                                              cv2.CHAIN_APPROX_SIMPLE)[0]
                    
                    # Draw filled contours on the overlay
                    cv2.fillPoly(overlay, contours, color)
                    
                    # Also draw contour outline
                    cv2.drawContours(annotated_frame, contours, -1, color, 2)
                    
                    # Add class label with confidence score
                    # Get box coordinates for placing the text
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    label = f"{class_name} {confidence:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Blend the overlay with the original frame
            alpha = 0.4  # Transparency factor
            cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
            
        elif current_mode == "pose":
            # Pose estimation mode
            results = models[current_mode](frame, conf=conf_threshold)
            
            # Get the first result
            result = results[0]
            
            # Process keypoints if available
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                for person_idx, kpts in enumerate(result.keypoints.data):
                    # Draw all keypoints for this person
                    for idx, (x, y, conf) in enumerate(kpts):
                        if conf > conf_threshold:  # Only draw keypoints above confidence threshold
                            x, y = int(x), int(y)
                            # Draw a circle for each keypoint
                            cv2.circle(annotated_frame, (x, y), 5, (0, 255, 0), -1)
                            
                            # Optionally show keypoint names
                            # cv2.putText(annotated_frame, keypoint_names[idx], (x+5, y+5), 
                            #            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                    
                    # Draw skeleton lines connecting keypoints
                    for i, (p1_idx, p2_idx) in enumerate(skeleton):
                        p1_idx -= 1  # Convert from 1-indexed to 0-indexed
                        p2_idx -= 1
                        # Only draw the line if both keypoints have sufficient confidence
                        # and are within the frame (not at 0,0 or invalid positions)
                        if (kpts[p1_idx][2] > conf_threshold and kpts[p2_idx][2] > conf_threshold and
                            kpts[p1_idx][0] > 1 and kpts[p1_idx][1] > 1 and 
                            kpts[p2_idx][0] > 1 and kpts[p2_idx][1] > 1):
                            
                            p1 = (int(kpts[p1_idx][0]), int(kpts[p1_idx][1]))
                            p2 = (int(kpts[p2_idx][0]), int(kpts[p2_idx][1]))
                            
                            # Additional safety check to avoid drawing lines to 0,0
                            if p1 != (0,0) and p2 != (0,0):
                                cv2.line(annotated_frame, p1, p2, (0, 255, 255), 2)
                            
                    # Get bounding box if available
                    if hasattr(result, 'boxes') and len(result.boxes) > person_idx:
                        box = result.boxes[person_idx]
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        confidence = float(box.conf[0])
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        label = f"Person {confidence:.2f}"
                        cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            

        
        # Display the current mode and model size in the corner of the frame
        mode_text = f"Mode: {current_mode.capitalize()} | Model: YOLOv8{current_size}"
        cv2.putText(annotated_frame, mode_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display FPS
        fps_text = f"FPS: {fps:.1f} | Conf: {conf_threshold:.2f}"
        cv2.putText(annotated_frame, fps_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Resize frame to the specified window dimensions if needed
        if annotated_frame.shape[1] != window_width or annotated_frame.shape[0] != window_height:
            annotated_frame = cv2.resize(annotated_frame, (window_width, window_height))
            
        # Display the result in a consistent window
        cv2.imshow('YOLOv8 Multi-Mode Webcam', annotated_frame)
    
    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()