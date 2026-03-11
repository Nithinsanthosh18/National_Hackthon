# Sentinel Vision
AI-powered surveillance video analysis system.

## Overview
Sentinel Vision detects security threats (weapons, intrusions, abandoned objects) in real-time using deep learning (Ultralytics YOLO) and OpenCV. It features a FastAPI backend logic and a modern React dashboard for operators to monitor live video feeds, real-time alerts, and spatial heatmaps.

## Prerequisites
- Python 3.9+
- Node.js & npm (v18+)

## Installation & Execution

### 1. Start the Backend
Open a terminal and run the following commands:
```bash
cd backend
# Create a virtual environment (optional but recommended)
python -m venv venv
# Activate the virtual environment
# Windows: venv\\Scripts\\activate
# macOS/Linux: source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Run the FastAPI server
python main.py
```
*Note: On first run, it will automatically download the `yolov8n.pt` weights file.*

### 2. Start the Frontend Dashboard
Open a new terminal and run:
```bash
cd frontend

# Install dependencies
npm install

# Start the Vite dev server
npm run dev
```
Navigate to `http://localhost:5173` in your browser to view the Sentinel Vision dashboard.

## Simulated Threats
By default, the system connects to your primary webcam (`source=0` in `main.py`). You can show objects like a backpack, suitcase, or person to testing. Moving a person into the "Restricted Zone" (red polygon) will trigger a High Alert.

## Project Structure
- **/backend**: FastAPI app, YOLO integration, SQLite database.
- **/frontend**: React + Vite application for real-time monitoring.
