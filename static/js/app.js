// --- Elements ---
const videoElement = document.getElementById('webcam');
const canvasElement = document.getElementById('overlay');
const resultsCanvasElement = document.getElementById('results-overlay');
const controlsPanel = document.getElementById('controls');
const statsPanel = document.getElementById('stats');
const detectionList = document.getElementById('detection-list');
const fpsDisplay = document.getElementById('fps-display');
const procTimeDisplay = document.getElementById('proc-time-display');
const currentModeDisplay = document.getElementById('current-mode-display');
const currentConfDisplay = document.getElementById('current-conf-display');
const videoSourceDisplay = document.getElementById('video-source-display');
const ctx = canvasElement.getContext('2d');
const resultsCtx = resultsCanvasElement.getContext('2d');

// --- Configuration ---
const motionStreamURL = "http://172.16.254.96:8082"; // URL of the Motion JPEG stream
const VIDEO_SOURCE = {
    WEBCAM: 'webcam',
    MOTION_STREAM: 'motionStream'
};

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
let currentVideoSource = VIDEO_SOURCE.WEBCAM; // Default to webcam
let webcamStream = null; // Store webcam MediaStream
let modeAutoChangeInterval = null;
let lastModeChangeTime = Date.now();

// --- Initialization ---
function init() {
    console.log("Initializing video source...");
    requestWebcamAccess(); // Start with webcam by default
    document.addEventListener('keydown', handleKeydown);
    updateModeDisplay();
    updateConfDisplay();
    updateVideoSourceDisplay();
    setInterval(updateStats, 1000); // Update stats display periodically
    startAutoModeCycling();
}

// --- Video Source Setup ---
// Webcam Setup
function requestWebcamAccess() {
    console.log("Requesting webcam access...");
    // Stop any existing streams
    stopCurrentVideoSource();
    
    // Set current source
    currentVideoSource = VIDEO_SOURCE.WEBCAM;
    updateVideoSourceDisplay();
    
    // Request webcam access
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            console.log("Webcam access granted");
            webcamStream = stream;
            videoElement.srcObject = stream;
            videoElement.style.display = 'block';
            
            // Wait for video to be ready
            videoElement.onloadedmetadata = () => {
                console.log("Webcam stream loaded");
                // Set canvas dimensions based on the video
                canvasElement.width = videoElement.videoWidth;
                canvasElement.height = videoElement.videoHeight;
                streamActive = true;
                lastFrameTime = performance.now(); // Reset FPS timer
                // Start processing
                requestAnimationFrame(processVideo);
            };
            
            videoElement.play();
        })
        .catch(error => {
            console.error("Error accessing webcam:", error);
            displayErrorOnCanvas("Could not access webcam", error.message);
            // Fall back to motion stream if webcam fails
            setTimeout(() => setupMotionStream(), 2000);
        });
}

// Motion Stream Setup
function setupMotionStream() {
    console.log("Setting up motion stream...");
    // Stop any existing streams
    stopCurrentVideoSource();
    
    // Set current source
    currentVideoSource = VIDEO_SOURCE.MOTION_STREAM;
    updateVideoSourceDisplay();
    
    // Hide video element
    if (videoElement) videoElement.style.display = 'none';
    
    // Setup motion stream
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
        streamActive = true;
        lastFrameTime = performance.now(); // Reset FPS timer
    }
    // Process the loaded frame
    processVideo();
}

function handleImageError() {
    console.error("Error loading Motion stream from:", motionStreamURL);
    streamActive = false;
    displayErrorOnCanvas("Error loading video stream", `Could not connect to ${motionStreamURL}`);

    // Attempt to reconnect after a delay
    console.log("Attempting reconnect in 5 seconds...");
    setTimeout(() => {
        if (!streamActive && currentVideoSource === VIDEO_SOURCE.MOTION_STREAM) { 
             motionImg.src = motionStreamURL + '?' + Date.now(); // Try again, busting cache
        }
    }, 5000);
}

function displayErrorOnCanvas(mainMessage, detailMessage) {
    ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    ctx.fillStyle = 'rgba(255, 0, 0, 0.7)';
    ctx.fillRect(0, 0, canvasElement.width, canvasElement.height);
    ctx.fillStyle = 'white';
    ctx.font = '20px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(mainMessage, canvasElement.width / 2, canvasElement.height / 2 - 15);
    ctx.font = '14px Arial';
    ctx.fillText(detailMessage, canvasElement.width / 2, canvasElement.height / 2 + 15);
}

function stopCurrentVideoSource() {
    streamActive = false;
    
    // Stop webcam if active
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
        videoElement.srcObject = null;
    }
    
    // Clear motion stream if active
    if (currentVideoSource === VIDEO_SOURCE.MOTION_STREAM) {
        motionImg.src = '';
    }
}

function toggleVideoSource() {
    if (currentVideoSource === VIDEO_SOURCE.WEBCAM) {
        setupMotionStream();
    } else {
        requestWebcamAccess();
    }
}

function startAutoModeCycling() {
    if (modeAutoChangeInterval === null) {
        console.log('Auto mode cycling started (every 10 seconds)');
        modeAutoChangeInterval = true;
        lastModeChangeTime = Date.now();
    }
}

function stopAutoModeCycling() {
    if (modeAutoChangeInterval !== null) {
        console.log('Auto mode cycling stopped');
        modeAutoChangeInterval = null;
    }
}

function toggleAutoModeCycling() {
    if (modeAutoChangeInterval === null) {
        startAutoModeCycling();
    } else {
        stopAutoModeCycling();
    }
}

// --- Main Processing Loop ---
async function processVideo() {
    if (!streamActive || isProcessing) {
        // If stream isn't active, error handling should reconnect.
        // If processing, wait for the current cycle to finish.
        if (currentVideoSource === VIDEO_SOURCE.WEBCAM && streamActive) {
            requestAnimationFrame(processVideo); // Continue loop for webcam
        }
        return;
    }

    isProcessing = true; // Set flag

    // Clear the main canvas before drawing the new frame
    ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    // Use a temporary canvas to get the image data
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    let sourceWidth, sourceHeight;

    // Handle different video sources
    if (currentVideoSource === VIDEO_SOURCE.WEBCAM) {
        // Check if video is ready
        if (videoElement.readyState < 2) { // HAVE_CURRENT_DATA or higher
            isProcessing = false;
            requestAnimationFrame(processVideo);
            return;
        }

        // Ensure canvas matches video dimensions
        if (canvasElement.width !== videoElement.videoWidth || canvasElement.height !== videoElement.videoHeight) {
            canvasElement.width = videoElement.videoWidth;
            canvasElement.height = videoElement.videoHeight;
            // Also resize the results canvas to match
            resultsCanvasElement.width = videoElement.videoWidth;
            resultsCanvasElement.height = videoElement.videoHeight;
            console.log("Canvases resized to:", canvasElement.width, canvasElement.height);
        }

        // Draw video frame to main canvas
        try {
            ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
        } catch (e) {
            console.error("Error drawing webcam video:", e);
            isProcessing = false;
            requestAnimationFrame(processVideo);
            return;
        }

        // Setup temp canvas for processing
        sourceWidth = videoElement.videoWidth;
        sourceHeight = videoElement.videoHeight;
        tempCanvas.width = sourceWidth;
        tempCanvas.height = sourceHeight;
        tempCtx.drawImage(videoElement, 0, 0, tempCanvas.width, tempCanvas.height);
    } else { // MOTION_STREAM
        // Ensure canvas matches image dimensions
        if (canvasElement.width !== motionImg.naturalWidth || canvasElement.height !== motionImg.naturalHeight) {
            canvasElement.width = motionImg.naturalWidth;
            canvasElement.height = motionImg.naturalHeight;
            console.log("Canvas resized to:", canvasElement.width, canvasElement.height);
        }

        // Draw motion frame to main canvas
        try {
            ctx.drawImage(motionImg, 0, 0, canvasElement.width, canvasElement.height);
        } catch (e) {
            console.error("Error drawing motion image:", e);
            isProcessing = false;
            handleImageError(); // Trigger error handling
            return;
        }

        // Setup temp canvas for processing
        sourceWidth = motionImg.naturalWidth;
        sourceHeight = motionImg.naturalHeight;
        tempCanvas.width = sourceWidth;
        tempCanvas.height = sourceHeight;
        tempCtx.drawImage(motionImg, 0, 0, tempCanvas.width, tempCanvas.height);
    }

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
        
        // Redraw the current video frame based on source
        if (currentVideoSource === VIDEO_SOURCE.WEBCAM) {
            ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
        } else {
            ctx.drawImage(motionImg, 0, 0, canvasElement.width, canvasElement.height);
        }

        // Log the received data for debugging
        console.log("Received data:", data);

        // Draw results onto the overlay canvas
        if (data.results) {
             // Pass original dimensions for scaling
             console.log("Drawing results!")
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
        // Display error overlay
        displayErrorOnCanvas('Error processing image', error.message);
        ctx.textAlign = 'center';
        ctx.fillText('Processing Error', canvasElement.width / 2, canvasElement.height / 2);

    } finally {
         isProcessing = false; // Reset flag
         // requestAnimationFrame(processVideo); // Old loop mechanism
         // Trigger loading the next frame from the stream
         if (streamActive) {
             if (currentVideoSource === VIDEO_SOURCE.WEBCAM) {
                 requestAnimationFrame(processVideo); // Continue loop for webcam
             } else {
                 motionImg.src = motionStreamURL + '?' + Date.now(); // Force reload by changing URL slightly
             }
         }
    }
}

// --- Drawing Functions ---

function getRandomColor(seed) {
    // Predefined color palette with distinct, vibrant colors
    const colorPalette = [
        'rgb(255, 0, 0)',      // Red
        'rgb(0, 255, 0)',      // Green
        'rgb(0, 0, 255)',      // Blue
        'rgb(255, 255, 0)',    // Yellow
        'rgb(255, 0, 255)',    // Magenta
        'rgb(0, 255, 255)',    // Cyan
        'rgb(255, 128, 0)',    // Orange
        'rgb(128, 0, 255)',    // Purple
        'rgb(0, 128, 255)',    // Light Blue
        'rgb(255, 0, 128)',    // Pink
        'rgb(128, 255, 0)',    // Lime
        'rgb(0, 255, 128)',    // Teal
        'rgb(128, 128, 255)',  // Lavender
        'rgb(255, 128, 128)',  // Light Red
        'rgb(128, 255, 128)',  // Light Green
        'rgb(192, 0, 0)',      // Dark Red
        'rgb(0, 192, 0)',      // Dark Green
        'rgb(0, 0, 192)',      // Dark Blue
        'rgb(192, 192, 0)',    // Olive
        'rgb(192, 0, 192)'     // Dark Magenta
    ];
    
    // Use the seed to select a color from the palette
    let index = 0;
    const strSeed = String(seed);
    for (let i = 0; i < strSeed.length; i++) {
        index = (index + strSeed.charCodeAt(i)) % colorPalette.length;
    }
    
    return colorPalette[index];
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
    // Make sure the results canvas is properly sized
    if (resultsCanvasElement.width !== canvasElement.width || resultsCanvasElement.height !== canvasElement.height) {
        resultsCanvasElement.width = canvasElement.width;
        resultsCanvasElement.height = canvasElement.height;
        console.log("Results canvas resized to:", resultsCanvasElement.width, resultsCanvasElement.height);
    }

    // Clear the results canvas before drawing
    resultsCtx.clearRect(0, 0, resultsCanvasElement.width, resultsCanvasElement.height);

    // Check if original dimensions are valid
    if (!originalWidth || !originalHeight || originalWidth <= 0 || originalHeight <= 0) {
        console.error("Invalid original dimensions received:", originalWidth, originalHeight);
        // Optionally draw an error message on canvas
        resultsCtx.fillStyle = 'red';
        resultsCtx.font = '16px Arial';
        resultsCtx.fillText('Error: Invalid dimensions from server.', 10, 20);
        return; // Stop drawing if dimensions are invalid
    }

    // Log mode and results count
    console.log(`drawResults called. Mode: ${currentMode}, Results: ${results.length}`);

    // Calculate scaling factors
    const scaleX = resultsCanvasElement.width / originalWidth;
    const scaleY = resultsCanvasElement.height / originalHeight;

    // No test elements needed anymore

    // Draw all results directly on the results canvas
    results.forEach((result, index) => {
        // Log the individual result object
    

        const { box, confidence, class_id, class_name, track_id, mask, keypoints, keypoints_conf } = result;
        
        // Get a unique color for this class
        const color = getColorForClass(class_id, class_name, track_id);
        
        // Set styles on results context
        resultsCtx.strokeStyle = color;
        resultsCtx.fillStyle = color;
        resultsCtx.lineWidth = 2;
        resultsCtx.font = '12px Arial';

        // --- Draw Bounding Box (only in detection mode) ---
        if (box && currentMode === 'detection') {
            const [x1, y1, x2, y2] = box;
            const scaledX1 = Math.round(x1 * scaleX);
            const scaledY1 = Math.round(y1 * scaleY);
            const scaledWidth = Math.round((x2 - x1) * scaleX);
            const scaledHeight = Math.round((y2 - y1) * scaleY);

            // Draw box with class-specific color
            resultsCtx.strokeStyle = color;
            resultsCtx.lineWidth = 3;
            resultsCtx.strokeRect(scaledX1, scaledY1, scaledWidth, scaledHeight);

            // Draw label with contrasting background
            let label = `${class_name} ${confidence.toFixed(2)}`;
            if (track_id !== null && track_id !== undefined) {
                label += ` #${track_id}`;
            }
            
            resultsCtx.fillStyle = 'black';
            resultsCtx.fillRect(scaledX1, scaledY1 - 20, resultsCtx.measureText(label).width + 10, 20);
            resultsCtx.fillStyle = 'white';
            resultsCtx.fillText(label, scaledX1 + 5, scaledY1 - 5);
        }
        
        // --- Draw Segmentation Mask ---
        if (mask && currentMode === 'segmentation') {
            resultsCtx.globalAlpha = 0.5; // Semi-transparent mask
            resultsCtx.fillStyle = color;
            resultsCtx.beginPath();
            
            mask.forEach((point, index) => {
                const scaledX = Math.round(point[0] * scaleX);
                const scaledY = Math.round(point[1] * scaleY);
                if (index === 0) {
                    resultsCtx.moveTo(scaledX, scaledY);
                } else {
                    resultsCtx.lineTo(scaledX, scaledY);
                }
            });
            
            resultsCtx.closePath();
            resultsCtx.fill();
            resultsCtx.globalAlpha = 1.0; // Reset alpha
        }
        
        // --- Draw Pose Keypoints ---
        if (keypoints && keypoints_conf && currentMode === 'pose') {
            // Define connections (pairs of keypoint indices)
            // COCO keypoint indices: 0:Nose, 1:LEye, 2:REye, 3:LEar, 4:REar, 5:LShoulder, 6:RShoulder, 7:LElbow, 8:RElbow, 9:LWrist, 10:RWrist, 11:LHip, 12:RHip, 13:LKnee, 14:RKnee, 15:LAnkle, 16:RAnkle
            const connections = [
                [0, 1], [0, 2], [1, 3], [2, 4], // Head
                [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], // Arms
                [11, 12], [5, 11], [6, 12], // Torso
                [11, 13], [13, 15], [12, 14], [14, 16] // Legs
            ];
            const keypointColor = color; // Use class color for keypoints
            const connectionColor = color;
            const keypointRadius = 3;
            const minKeypointConf = 0.5; // Minimum confidence to draw a keypoint/connection

            // Draw connections
            resultsCtx.strokeStyle = connectionColor;
            resultsCtx.lineWidth = 2;
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

                    resultsCtx.beginPath();
                    resultsCtx.moveTo(scaledX1_kp, scaledY1_kp);
                    resultsCtx.lineTo(scaledX2_kp, scaledY2_kp);
                    resultsCtx.stroke();
                }
            });

            // Draw keypoints
            resultsCtx.fillStyle = keypointColor;
            keypoints.forEach((point, index) => {
                if (keypoints_conf.length > index && keypoints_conf[index] >= minKeypointConf) {
                    const [x, y] = point;
                    const scaledX = Math.round(x * scaleX);
                    const scaledY = Math.round(y * scaleY);
                    resultsCtx.beginPath();
                    resultsCtx.arc(scaledX, scaledY, keypointRadius, 0, 2 * Math.PI);
                    resultsCtx.fill();
                }
            });
        }
    });
    
    // No final marker needed anymore
    
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
    
    // Check if it's time to auto-change the mode
    if (modeAutoChangeInterval !== null && Date.now() - lastModeChangeTime > 10000) {
        cycleMode();
        lastModeChangeTime = Date.now();
    }
    
    // Schedule next update
    setTimeout(updateStats, 1000);
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
        case 'V':
            toggleVideoSource();
            break;
        case 'ARROWUP':
             adjustConfidence(0.05);
             break;
        case 'ARROWDOWN':
             adjustConfidence(-0.05);
             break;
        case 'A':
            toggleAutoModeCycling();
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

function updateVideoSourceDisplay() {
    if (videoSourceDisplay) {
        videoSourceDisplay.textContent = currentVideoSource === VIDEO_SOURCE.WEBCAM ? 'Webcam' : 'Motion Stream';
    }
}


// --- Start ---
init();
