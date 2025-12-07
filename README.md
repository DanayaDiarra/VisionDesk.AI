# VisionDesk.AI - Computer Vision Suite

A comprehensive multi-tool computer vision application built with Streamlit and YOLOv11.

![VisionDesk.AI](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)

## ✨ Features
- **Real-time Object Detection** using YOLOv11
- **Image Captioning** with AI models
- **Face Detection & Recognition**
- **Image Classification** across multiple categories
- **Text Extraction (OCR)** from images
- **Gallery Management** with search and filter
- **Logging & Analytics** dashboard

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/DanayaDiarra/VisionDesk.AI.git
cd VisionDesk.AI
pip install -r requirements.txt


📁 Project Structure
VisionDesk.AI/
├── Main_page.py          # Main application entry
├── stapp.py             # Alternative app entry
├── config.json          # Application configuration
├── requirements.txt     # Python dependencies
├── .streamlit/          # Streamlit configuration
│   └── config.toml
├── pages/               # Feature modules (multi-page)
│   ├── 1_Caption_Images.py
│   ├── 2_Detect_Objects.py
│   ├── 3_Detect_Faces.py
│   ├── 4_Classify_Images.py
│   ├── 5_Extract_Text.py
│   ├── 5_Faces_database.py
│   ├── 6_Gallery.py
│   └── 7_Logs_and_stats.py
├── yolo11n.pt           # YOLO model weights
├── runnn.ipynb          # Jupyter notebook for experiments
└── history.log          # Application logs
