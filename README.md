# Automated Road Damage Detection System

## Overview

An end-to-end deep learning–based road damage detection system built using YOLOv8, Streamlit, and computer vision techniques.

This project detects multiple types of road damage such as:

* Longitudinal cracks
* Transverse cracks
* Alligator cracks
* Potholes
* Unknown road damages

The system supports:

* Image-based detection
* Real-time webcam detection
* Video inference
* Interactive Streamlit web application

---

# Project Demo

## Features

* YOLOv8 object detection pipeline
* Real-time webcam inference
* Streamlit AI web application
* GPU training using Kaggle
* Dataset preprocessing and analysis
* Annotation visualization
* Detection confidence scoring
* Prediction result visualization
* Organized modular project structure

---

# Technologies Used

## Deep Learning & Computer Vision

* Python
* YOLOv8
* OpenCV
* PyTorch
* Ultralytics

## Web Application

* Streamlit

## Data Processing

* NumPy
* Matplotlib
* Pillow

## Development Tools

* VS Code
* Git & GitHub
* Kaggle GPU

---

# Dataset

Dataset used:
RDD2022 (Road Damage Detection Dataset)

The dataset contains annotated road images collected from multiple countries and road environments.

Damage Classes:

* D00 — Longitudinal Crack
* D10 — Transverse Crack
* D20 — Alligator Crack
* D40 — Pothole
* Unknown Damage

---

# Project Structure

```bash
Automated-Road-Damage-Detection-System/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│
├── models/
│   └── baseline/
│
├── results/
│   ├── predictions/
│   └── training_curves/
│
├── src/
│   ├── inference/
│   ├── preprocessing/
│   └── utils/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Model Training

Baseline model:

* YOLOv8n
* Trained on Kaggle Tesla T4 GPU
* Input size: 640x640
* Epochs: 5 (baseline training)

Future improvements:

* 50+ epoch training
* Hyperparameter tuning
* Model optimization
* Advanced YOLO architectures

---

# Training Results

## Baseline Performance

| Metric    | Score |
| --------- | ----- |
| Precision | ~0.47 |
| Recall    | ~0.43 |
| mAP50     | ~0.41 |
| mAP50-95  | ~0.20 |

---

# Streamlit Web Application

The project includes an interactive web application where users can:

* Upload road images
* Detect road damage
* Visualize predictions
* Adjust confidence threshold

Run locally:

```bash
streamlit run app/streamlit_app.py
```

---

# Real-Time Detection

Supports:

* Webcam inference
* Video inference
* Live detection visualization

Run webcam detection:

```bash
python src/inference/webcam_inference.py
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/Bilal-Afzal-AI/automated-road-damage-detection-system.git
```

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Environment

### Windows

```bash
venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Inference

## Image Inference

```bash
python src/inference/predict_image.py
```

## Video Inference

```bash
python src/inference/predict_video.py
```

## Webcam Inference

```bash
python src/inference/webcam_inference.py
```

---

# Future Improvements

* Better model optimization
* More training epochs
* YOLOv8s / YOLOv9 experimentation
* Model deployment
* Docker support
* REST API integration
* Mobile deployment
* Edge AI optimization

---

# Author

Bilal Mohammad Afzal

* GitHub: [https://github.com/Bilal-Afzal-AI](https://github.com/Bilal-Afzal-AI)
* LinkedIn: Add your LinkedIn profile here

---

# License

This project is for educational and research purposes.
