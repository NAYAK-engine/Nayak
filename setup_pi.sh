#!/bin/bash

set -e

echo "================================"
echo "  NAYAK — Raspberry Pi Setup"
echo "================================"

# Step 1 — System update
echo "[1/7] Updating system..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Step 2 — Install Python 3.12
echo "[2/7] Installing Python 3.12..."
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev

# Step 3 — Install system dependencies
echo "[3/7] Installing system dependencies..."
sudo apt-get install -y \
    git \
    cmake \
    build-essential \
    libopencv-dev \
    python3-opencv \
    libatlas-base-dev \
    libjpeg-dev \
    libpng-dev \
    i2c-tools \
    python3-smbus \
    libgstreamer1.0-dev

# Step 4 — Enable camera and GPIO
echo "[4/7] Enabling camera and GPIO..."
sudo raspi-config nonint do_camera 0
sudo raspi-config nonint do_i2c 0
sudo raspi-config nonint do_spi 0

# Step 5 — Install NAYAK
echo "[5/7] Installing NAYAK..."
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install RPi.GPIO
pip install opencv-python
pip install picamera2

# Step 6 — Install Ollama for Pi
echo "[6/7] Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2

# Step 7 — Configure environment
echo "[7/7] Configuring NAYAK for Raspberry Pi..."
cp .env.example .env
sed -i 's/NAYAK_PROVIDER=ollama/NAYAK_PROVIDER=ollama/' .env
sed -i 's/NAYAK_DEVICE=desktop/NAYAK_DEVICE=raspberry-pi/' .env

echo ""
echo "================================"
echo "  NAYAK installed successfully!"
echo "  Run: python -m nayak run 'your goal'"
echo "================================"
