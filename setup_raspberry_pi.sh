#!/bin/bash
# Setup script for Raspberry Pi with Hailo-8L accelerator

echo "Setting up Party Cam application for Raspberry Pi with Hailo-8L..."

# Create directory for Hailo models
mkdir -p hailo_models

# Update system
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Python dependencies
echo "Installing Python dependencies..."
sudo apt-get install -y python3-pip python3-opencv python3-numpy python3-pillow

# Install Flask and Waitress
echo "Installing web server dependencies..."
pip3 install flask waitress

# Install Hailo SDK
echo "Installing Hailo SDK..."
# Note: The actual installation commands may vary based on Hailo's latest documentation
# This is a placeholder - refer to Hailo's official documentation for the exact commands
echo "Please follow Hailo's official documentation to install the Hailo SDK"
echo "Typically, this involves:"
echo "1. Adding Hailo's package repository"
echo "2. Installing hailo-ai package"
echo "3. Setting up udev rules for the Hailo device"

# Create a directory for Hailo models
mkdir -p hailo_models
echo "You'll need to place your Hailo model files (.hef) in the hailo_models directory"
echo "For YOLOv5s models, you'll need:"
echo "- hailo_models/yolov5s.hef (detection)"
echo "- hailo_models/yolov5s-seg.hef (segmentation)"
echo "- hailo_models/yolov5s-pose.hef (pose estimation)"

# Set up autostart (optional)
echo "Would you like to set up the application to start automatically on boot? (y/n)"
read autostart

if [ "$autostart" = "y" ]; then
    echo "Setting up autostart..."
    
    # Create service file
    cat > party-cam.service << EOF
[Unit]
Description=Party Cam Application
After=network.target

[Service]
ExecStart=/usr/bin/python3 $(pwd)/app_hailo.py
WorkingDirectory=$(pwd)
Restart=always
User=$USER

[Install]
WantedBy=multi-user.target
EOF

    # Move service file to systemd directory
    sudo mv party-cam.service /etc/systemd/system/
    
    # Enable and start the service
    sudo systemctl enable party-cam.service
    sudo systemctl start party-cam.service
    
    echo "Autostart configured. The application will start automatically on boot."
    echo "You can control the service with:"
    echo "sudo systemctl start party-cam.service"
    echo "sudo systemctl stop party-cam.service"
    echo "sudo systemctl restart party-cam.service"
fi

echo ""
echo "Setup complete!"
echo "To run the application manually:"
echo "python3 app_hailo.py"
echo ""
echo "The web interface will be available at:"
echo "http://[raspberry-pi-ip]:5100"
echo ""
echo "Note: Make sure your Hailo-8L is properly connected and the models are in the hailo_models directory."
