# Real-Time Face Recognition System

A Python-based real-time face detection and recognition application using a laptop webcam.

Built using OpenCV, YuNet, SFace and CustomTkinter.

## Features

- Real-time face detection.
- Multiple-face detection.
- Register people using five different facial poses.
- Recognise registered people in real time.
- Identify unregistered faces as Unknown.
- Store face embeddings locally.
- Modern desktop interface.
- CPU-based processing.
- No paid APIs or cloud services required.

## Technologies Used

- Python
- OpenCV
- NumPy
- YuNet
- SFace
- ONNX
- CustomTkinter
- Pillow
- JSON

## AI Models

### YuNet – Face Detection
- Detects faces and facial landmarks.
- Source: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- License: MIT

### SFace – Face Recognition
- Generates numerical face embeddings.
- Architecture: MobileFaceNet
- Source: https://huggingface.co/opencv/face_recognition_sface
- License: Apache 2.0

Both models run locally through OpenCV.

## How It Works

1. The webcam captures live video.
2. YuNet detects faces.
3. OpenCV aligns and crops detected faces.
4. SFace generates a 128-dimensional face embedding.
5. Cosine similarity compares the embedding with registered faces.
6. The application identifies the person or displays Unknown.

## Face Registration

People can be registered using five facial poses:

- Front
- Left
- Right
- Up
- Down

The resulting embeddings are stored locally in a JSON database.

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd Real-Time-Face-Recognition
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
python app.py
```

## Privacy

Registered face embeddings are stored locally. Personal face databases and photographs are excluded from the public repository.

## Limitations

This is an educational computer vision project, not a security-grade authentication system. Recognition performance may vary depending on lighting, camera quality and face angles.

## Model Licenses

- YuNet: MIT License.
- SFace: Apache 2.0 License.

See the `licenses/` directory for the original licence texts.

## Project Purpose

This project explores real-time computer vision, face detection, deep-learning-based face recognition, embeddings and similarity matching using locally running AI models.