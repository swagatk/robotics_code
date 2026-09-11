#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=== Step 1: Updating system and installing build tools ==="
sudo apt update
sudo apt upgrade -y
sudo apt install -y python3-pip wget

# Install vcstool and colcon using pip
sudo pip install --break-system-packages vcstool colcon-common-extensions

echo "=== Step 2: Downloading ROS2 Jazzy package from Google Drive ==="
mkdir -p ~/Downloads
cd ~/Downloads

FILE_ID="1FMl0GQ9G3Z3eYjtIaxLWyWn5Arg0J06U"
OUTPUT_FILE="ros-jazzy-desktop-0.3.2_20240525_arm64.deb"

# Fetch download confirmation token for large Google Drive files and download
CONFIRM_TOKEN=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
  "https://docs.google.com/uc?export=download&id=${FILE_ID}" -O- | \
  sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')

if [ -n "$CONFIRM_TOKEN" ]; then
  wget --load-cookies /tmp/cookies.txt \
    "https://docs.google.com/uc?export=download&confirm=${CONFIRM_TOKEN}&id=${FILE_ID}" \
    -O "$OUTPUT_FILE"
else
  wget --no-check-certificate \
    "https://docs.google.com/uc?export=download&id=${FILE_ID}" \
    -O "$OUTPUT_FILE"
fi

rm -f /tmp/cookies.txt

echo "=== Step 3: Installing ROS2 Jazzy .deb package ==="
# apt install handles local .deb files along with missing dependencies
sudo apt install -y "./${OUTPUT_FILE}"

echo "=== Step 4: Setting up ROS environment in ~/.bashrc ==="
ROS_SETUP="source /opt/ros/jazzy/setup.bash"
ROS_DOMAIN="export ROS_DOMAIN_ID=19"

# Append only if not already present to avoid duplicate lines
grep -qxF "$ROS_SETUP" ~/.bashrc || echo "$ROS_SETUP" >> ~/.bashrc
grep -qxF "$ROS_DOMAIN" ~/.bashrc || echo "$ROS_DOMAIN" >> ~/.bashrc

echo "=== Step 5: Verification ==="
# Source directly in current subshell to test variables without opening nested shells
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=19

printenv | grep ROS

echo ""
echo "Installation complete. Run 'source ~/.bashrc' or open a new terminal to start using ROS 2 Jazzy."