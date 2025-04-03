// --- Elements ---
const videoElement = document.getElementById('webcam'); // Keep for potential future use? Or remove if definitely not needed.
const canvasElement = document.getElementById('overlay');
const controlsPanel = document.getElementById('controls');
const statsPanel = document.getElementById('stats');
const detectionList = document.getElementById('detection-list');
const fpsDisplay = document.getElementById('fps-display');
const procTimeDisplay = document.getElementById('proc-time-display');
const currentModeDisplay = document.getElementById('current-mode-display');
const currentConfDisplay = document.getElementById('current-conf-display');
const ctx = canvasElement.getContext('2d');

// --- Configuration ---
const motionStreamURL = "http://172.16.254.96:8082"; // URL of the Motion JPEG stream

// --- State ---
let currentMode = initialMode; // From Jinja
let currentSize = initialSize; // From Jinja
let currentConf = initialConf; // From Jinja
let streamActive = false;
let lastFrameTime = performance.now();
let frameCount = 0;
let fps = 0;
let isProcessing = false; // Flag to prevent concurrent requests
const classColors = {}; // Cache for generated colors
let motionImg = new Image(); // Image object for the Motion stream

// --- Initialization ---
function init() {
    console.log("Initializing Motion stream...");
    // requestWebcamAccess(); // Removed webcam access
    setupMotionStream();
    document.addEventListener('keydown', handleKeydown);
    updateModeDisplay();
    updateConfDisplay();
    setInterval(updateStats, 1000); // Update stats display periodically
}

// --- Motion Stream Setup ---
function setupMotionStream() {
    motionImg.crossOrigin = 'anonymous';
    motionImg.onload = handleImageLoad;
    motionImg.onerror = handleImageError;
    console.log("Attempting to load Motion stream:", motionStreamURL);
    motionImg.src = motionStreamURL; // Start loading
}

function handleImageLoad() {
    // console.log("Motion frame loaded."); // Debugging: can be noisy
    if (!streamActive) {
        console.log("Motion stream active.");
        // Set canvas dimensions based on the first frame
        canvasElement.width = motionImg.naturalWidth;
        canvasElement.height = motionImg.naturalHeight;
        // Hide the original video element if it's still there
        if (videoElement) videoElement.style.display = 'none';
        streamActive = true;
        lastFrameTime = performance.now(); // Reset FPS timer
    }
    // Process the loaded frame
    processVideo();
}

function handleImageError() {
    console.error("Error loading Motion stream from:", motionStreamURL);
    streamActive = false;
    // Display error on canvas
    ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    ctx.fillStyle = 'rgba(255, 0, 0, 0.7)';
    ctx.fillRect(0, 0, canvasElement.width, canvasElement.height);
    ctx.fillStyle = 'white';
    ctx.font = '20px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Error loading video stream.', canvasElement.width / 2, canvasElement.height / 2 - 15);
    ctx.font = '14px Arial';
    ctx.fillText(`Could not connect to ${motionStreamURL}`, canvasElement.width / 2, canvasElement.height / 2 + 15);

    // Attempt to reconnect after a delay
    console.log("Attempting reconnect in 5 seconds...");
    setTimeout(() => {
        if (!streamActive) { // Only try if still not active
             motionImg.src = motionStreamURL + '?' + Date.now(); // Try again, busting cache
        }
    }, 5000);
}


// --- Main Processing Loop ---
async function processVideo() {
    // Use motionImg instead of videoElement
    // if (!streamActive || videoElement.paused || videoElement.ended || isProcessing) { // Old check
    //     requestAnimationFrame(processVideo); // Keep looping even if paused/processing
    //     return;
    // }
    if (!streamActive || isProcessing) {
        // If stream isn't active, error handling should reconnect.
        // If processing, wait for the current cycle to finish.
        // The loop is continued by handleImageLoad -> processVideo -> motionImg.src reload
        return;
    }

    // Check if video is ready to capture frame (No longer needed for image)
    /*
    if (videoElement.readyState < 2) { // HAVE_CURRENT_DATA or higher
        requestAnimationFrame(processVideo);
        return;
    }
    */

    isProcessing = true; // Set flag

    // Ensure canvas matches image dimensions (in case it changes?) - unlikely for motion stream
    if (canvasElement.width !== motionImg.naturalWidth || canvasElement.height !== motionImg.naturalHeight) {
         canvasElement.width = motionImg.naturalWidth;
         canvasElement.height = motionImg.naturalHeight;
         console.log("Canvas resized to:", canvasElement.width, canvasElement.height); // Log resize
    }

    // Clear the main canvas *before* drawing the new frame
    ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    // Draw the current motion frame onto the main canvas (for user to see)
    try {
        ctx.drawImage(motionImg, 0, 0, canvasElement.width, canvasElement.height);
    } catch (e) {
         console.error("Error drawing motion image:", e);
         isProcessing = false;
         // Don't request next frame immediately if drawing failed
         handleImageError(); // Trigger error handling
         return;
    }


    // Use a temporary canvas to get the image data
    const tempCanvas = document.createElement('canvas');
    // Use natural dimensions from the image for the temp canvas
    tempCanvas.width = motionImg.naturalWidth;
    tempCanvas.height = motionImg.naturalHeight;
    const tempCtx = tempCanvas.getContext('2d');
    // Draw the motion image onto the temporary canvas
    tempCtx.drawImage(motionImg, 0, 0, tempCanvas.width, tempCanvas.height);

    // Get image data as base64
    const imageData = tempCanvas.toDataURL('image/jpeg', 0.8); // Use JPEG, quality 0.8

    try {
        const startTime = performance.now();
        const response = await fetch('/process_image', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        const endTime = performance.now();
        // console.log("Received data:", data); // Debugging

        // Get original dimensions from backend
        const { original_width, original_height } = data;

        // We need to redraw the video frame and then add the annotations
        // First, clear the canvas
        ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        
        // Redraw the current motion frame
        ctx.drawImage(motionImg, 0, 0, canvasElement.width, canvasElement.height);

        // Draw results onto the overlay canvas
        if (data.results) {
             // Pass original dimensions for scaling
             drawResults(data.results, original_width, original_height);
             // Update server processing time display immediately
             procTimeDisplay.textContent = data.processing_time_ms?.toFixed(1) ?? 'N/A';
        }

         // Calculate Client FPS
        const now = performance.now();
        frameCount++;
        const delta = now - lastFrameTime;
        if (delta >= 1000) { // Update FPS every second
            fps = (frameCount * 1000) / delta;
            fpsDisplay.textContent = fps.toFixed(1);
            lastFrameTime = now;
            frameCount = 0;
        }


    } catch (error) {
        console.error('Error processing image:', error);
        // Optionally clear canvas or display an error overlay
        ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        ctx.fillStyle = 'rgba(255, 0, 0, 0.5)';
        ctx.fillRect(0, 0, canvasElement.width, canvasElement.height);
        ctx.fillStyle = 'white';
        ctx.font = '20px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Processing Error', canvasElement.width / 2, canvasElement.height / 2);

    } finally {
         isProcessing = false; // Reset flag
         // requestAnimationFrame(processVideo); // Old loop mechanism
         // Trigger loading the next frame from the stream
         if (streamActive) {
             motionImg.src = motionStreamURL + '?' + Date.now(); // Force reload by changing URL slightly
         }
    }
}

// --- Drawing Functions ---

function getRandomColor(seed) {
    // Simple hash function for seed -> color mapping
    let hash = 0;
    const strSeed = String(seed);
    for (let i = 0; i < strSeed.length; i++) {
        hash = strSeed.charCodeAt(i) + ((hash << 5) - hash);
        hash = hash & hash; // Convert to 32bit integer
    }
    const r = (hash & 0xFF0000) >> 16;
    const g = (hash & 0x00FF00) >> 8;
    const b = hash & 0x0000FF;
    return `rgb(${r}, ${g}, ${b})`;
}

function getColorForClass(classId, className, trackId = null) {
    const key = trackId !== null ? `track_${trackId}` : `class_${classId}`;
    if (!classColors[key]) {
        if (trackId !== null) {
            // Generate color based on track ID for tracking consistency
             classColors[key] = getRandomColor(trackId);
        } else {
            // Generate color based on class ID for detection/segmentation/pose
             classColors[key] = getRandomColor(classId);
        }
    }
    return classColors[key];
}


function drawResults(results, originalWidth, originalHeight) {
    // Check if original dimensions are valid
    if (!originalWidth || !originalHeight || originalWidth <= 0 || originalHeight <= 0) {
        console.error("Invalid original dimensions received:", originalWidth, originalHeight);
        // Optionally draw an error message on canvas
        ctx.fillStyle = 'red';
        ctx.font = '16px Arial';
        ctx.fillText('Error: Invalid dimensions from server.', 10, 20);
        return; // Stop drawing if dimensions are invalid
    }
    
    console.log("Drawing results:", results.length, "items");

    // Calculate scaling factors
    const scaleX = canvasElement.width / originalWidth;
    const scaleY = canvasElement.height / originalHeight;

    // Draw all results directly on the main canvas
    results.forEach(result => {
        const { box, confidence, class_id, class_name, track_id, mask, keypoints, keypoints_conf } = result;
        //const [x1, y1, x2, y2] = box.map(coord => Math.round(coord)); // OLD: Direct mapping
        const color = getColorForClass(class_id, class_name, track_id);
        
        // Set styles on main context
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = 2;
        ctx.font = '12px Arial';

        // Styles are now set at the start of each result loop

        // --- Scale and Draw Bounding Box (Common to most modes) ---
        if (box && (currentMode === 'detection' || currentMode === 'tracking' || currentMode === 'pose' || currentMode === 'segmentation')) { // Draw box for seg/pose too for label
            const [x1, y1, x2, y2] = box;
            const scaledX1 = Math.round(x1 * scaleX);
            const scaledY1 = Math.round(y1 * scaleY);
            const scaledWidth = Math.round((x2 - x1) * scaleX);
            const scaledHeight = Math.round((y2 - y1) * scaleY);

            ctx.strokeRect(scaledX1, scaledY1, scaledWidth, scaledHeight);

            // --- Draw Label (using scaled coordinates) ---
            let label = `${class_name}`;
            if (track_id !== null && track_id !== undefined) {
                 label += ` #${track_id}`;
            }
            label += ` ${confidence.toFixed(2)}`;

            const textWidth = ctx.measureText(label).width;
            // Position label relative to the scaled box
            ctx.fillRect(scaledX1, scaledY1 - 14, textWidth + 4, 14); // Background for text
            ctx.fillStyle = 'white'; // Text color
            ctx.fillText(label, scaledX1 + 2, scaledY1 - 2);
            ctx.fillStyle = color; // Reset fillStyle for next element
        }

        // --- Scale and Draw Segmentation Mask ---
        if (mask && currentMode === 'segmentation') {
            ctx.globalAlpha = 0.5; // Semi-transparent mask
            ctx.beginPath();
            mask.forEach((point, index) => {
                const scaledX = Math.round(point[0] * scaleX);
                const scaledY = Math.round(point[1] * scaleY);
                if (index === 0) {
                    ctx.moveTo(scaledX, scaledY);
                } else {
                    ctx.lineTo(scaledX, scaledY);
                }
            });
            ctx.closePath();
            ctx.fill();
            ctx.globalAlpha = 1.0; // Reset alpha
        }

        // --- Scale and Draw Pose Keypoints ---
        if (keypoints && keypoints_conf && currentMode === 'pose') {
            // Define connections (pairs of keypoint indices)
            // COCO keypoint indices: 0:Nose, 1:LEye, 2:REye, 3:LEar, 4:REar, 5:LShoulder, 6:RShoulder, 7:LElbow, 8:RElbow, 9:LWrist, 10:RWrist, 11:LHip, 12:RHip, 13:LKnee, 14:RKnee, 15:LAnkle, 16:RAnkle
            const connections = [
                [0, 1], [0, 2], [1, 3], [2, 4], // Head
                [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], // Arms
                [11, 12], [5, 11], [6, 12], // Torso
                [11, 13], [13, 15], [12, 14], [14, 16] // Legs
            ];
            const keypointColor = 'lime'; // Use a distinct color for keypoints
            const connectionColor = 'cyan';
            const keypointRadius = 3;
            const minKeypointConf = 0.5; // Minimum confidence to draw a keypoint/connection

            // Draw connections
            ctx.strokeStyle = connectionColor;
            ctx.lineWidth = 1;
            connections.forEach(conn => {
                const [idx1, idx2] = conn;
                if (keypoints.length > idx1 && keypoints.length > idx2 &&
                    keypoints_conf.length > idx1 && keypoints_conf.length > idx2 &&
                    keypoints_conf[idx1] >= minKeypointConf && keypoints_conf[idx2] >= minKeypointConf) {

                    const [x_1, y_1] = keypoints[idx1];
                    const [x_2, y_2] = keypoints[idx2];
                    const scaledX1_kp = Math.round(x_1 * scaleX);
                    const scaledY1_kp = Math.round(y_1 * scaleY);
                    const scaledX2_kp = Math.round(x_2 * scaleX);
                    const scaledY2_kp = Math.round(y_2 * scaleY);

                    ctx.beginPath();
                    ctx.moveTo(scaledX1_kp, scaledY1_kp);
                    ctx.lineTo(scaledX2_kp, scaledY2_kp);
                    ctx.stroke();
                }
            });

            // Draw keypoints
            ctx.fillStyle = keypointColor;
            keypoints.forEach((point, index) => {
                if (keypoints_conf.length > index && keypoints_conf[index] >= minKeypointConf) {
                    const [x, y] = point;
                    const scaledX = Math.round(x * scaleX);
                    const scaledY = Math.round(y * scaleY);
                    ctx.beginPath();
                    ctx.arc(scaledX, scaledY, keypointRadius, 0, 2 * Math.PI);
                    ctx.fill();
                }
            });
        }
    });
    
    // Debug - confirm drawing completed
    console.log("Drawing completed directly on canvas");
}

// --- Stats Update ---
async function updateStats() {
    try {
        const response = await fetch('/get_stats');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();

        // Update detection list
        detectionList.innerHTML = ''; // Clear previous list
        if (data.detections && Object.keys(data.detections).length > 0) {
            for (const [className, count] of Object.entries(data.detections)) {
                const listItem = document.createElement('li');
                listItem.textContent = `${className}: ${count}`;
                detectionList.appendChild(listItem);
            }
        } else {
             const listItem = document.createElement('li');
             listItem.textContent = 'None';
             detectionList.appendChild(listItem);
        }

        // Note: FPS is updated in processVideo loop
        // Note: Server Proc Time is updated in processVideo loop


    } catch (error) {
        console.error("Error fetching stats:", error);
        detectionList.innerHTML = '<li>Error loading stats</li>';
    }
}


// --- Control Handlers ---
function handleKeydown(event) {
    switch (event.key.toUpperCase()) {
        case 'H': // Changed from 'C'
            controlsPanel.classList.toggle('hidden');
            break;
        case 'S':
            statsPanel.classList.toggle('hidden');
            break;
        case 'M':
            cycleMode();
            break;
        case 'ARROWUP':
             adjustConfidence(0.05);
             break;
        case 'ARROWDOWN':
             adjustConfidence(-0.05);
             break;
    }
}

function cycleMode() {
    const currentIndex = availableModes.indexOf(currentMode);
    const nextIndex = (currentIndex + 1) % availableModes.length;
    setMode(availableModes[nextIndex]);
}

function adjustConfidence(delta) {
     let newConf = parseFloat((currentConf + delta).toFixed(2)); // Keep 2 decimal places
     newConf = Math.max(0.05, Math.min(1.0, newConf)); // Clamp between 0.05 and 1.0
     setConfidence(newConf);
}


async function setMode(mode) {
    if (!availableModes.includes(mode)) {
        console.error("Invalid mode:", mode);
        return;
    }
    console.log("Setting mode to:", mode);
    try {
         // Log the request details before sending
         console.log(`Sending request to /set_mode with mode: ${mode}, size: ${currentSize}`);
         const response = await fetch('/set_mode', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: mode, size: currentSize }) // Send current size too
        });
        if (!response.ok) throw new Error(`Server error: ${response.status}`);
        const data = await response.json();
        if (data.status === 'success') {
            currentMode = data.mode;
            updateModeDisplay();
            classColors = {}; // Clear color cache on mode change
            console.log("Mode successfully set to:", currentMode);
        } else {
            console.error("Failed to set mode:", data.message);
        }
    } catch (error) {
        console.error("Error setting mode:", error);
    }
}

async function setConfidence(conf) {
     const newConf = parseFloat(conf.toFixed(2));
     console.log("Setting confidence to:", newConf);
     try {
        const response = await fetch('/set_conf', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ conf: newConf })
        });
         if (!response.ok) throw new Error(`Server error: ${response.status}`);
         const data = await response.json();
         if (data.status === 'success') {
             currentConf = data.conf;
             updateConfDisplay();
             console.log("Confidence successfully set to:", currentConf);
         } else {
             console.error("Failed to set confidence:", data.message);
         }
    } catch (error) {
        console.error("Error setting confidence:", error);
    }
}


// --- UI Updates ---
function updateModeDisplay() {
    currentModeDisplay.textContent = currentMode;
}

function updateConfDisplay() {
    currentConfDisplay.textContent = currentConf.toFixed(2);
}


// --- Start ---
init();
