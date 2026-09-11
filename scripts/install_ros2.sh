#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=== Step 1: Updating system and installing build tools ==="
sudo apt update
sudo apt upgrade -y
sudo apt install -y python3-pip

# Install vcstool, colcon, and gdown (to reliably fetch from Google Drive)
sudo pip install --break-system-packages vcstool colcon-common-extensions gdown

echo "=== Step 2: Downloading ROS2 Jazzy package using gdown ==="
mkdir -p ~/Downloads
cd ~/Downloads

FILE_ID="1FMl0GQ9G3Z3eYjtIaxLWyWn5Arg0J06U"
OUTPUT_FILE="ros-jazzy-desktop-0.3.2_20240525_arm64.deb"

# Remove any failed HTML download from previous attempts
rm -f "$OUTPUT_FILE"

# Download using file ID
gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "$OUTPUT_FILE"

# Verify that the downloaded file is indeed a Debian package and not HTML text
if file "$OUTPUT_FILE" | grep -q "Debian binary package"; then
    echo "Download verified successfully ($(du -h "$OUTPUT_FILE" | cut -f1))."
else
    echo "Error: Downloaded file is not a valid .deb package!"
    file "$OUTPUT_FILE"
    exit 1
fi

echo "=== Step 3: Installing ROS2 Jazzy .deb package ==="
# apt handles missing dependencies automatically
sudo apt install -y "./${OUTPUT_FILE}"

echo "=== Step 4: Setting up ROS environment in ~/.bashrc ==="
ROS_SETUP="source /opt/ros/jazzy/setup.bash"
ROS_DOMAIN="export ROS_DOMAIN_ID=19"

# Append only if not already present
grep -qxF "$ROS_SETUP" ~/.bashrc || echo "$ROS_SETUP" >> ~/.bashrc
grep -qxF "$ROS_DOMAIN" ~/.bashrc || echo "$ROS_DOMAIN" >> ~/.bashrc

echo "=== Step 5: Verification ==="
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=19

printenv | grep ROS

echo ""
echo "Installation complete. Run 'source ~/.bashrc' or restart your terminal."