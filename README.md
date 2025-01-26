# Coincent-Project-Task-2

Here’s a more concise and code-specific version of the README file in a clear, professional tone suitable for documentation:

---

# Face Mask Detection using OpenCV

## Overview
This project uses OpenCV and a pre-trained TensorFlow model to detect whether individuals in a video are wearing a face mask or not. It processes video frames, detects faces, and classifies them into "Mask" or "No Mask" categories.

---

## How It Works
1. **Face Detection**: 
   - The `haarcascade_frontalface_default.xml` file is used for detecting faces in frames.
   - Detected faces are extracted as individual regions.

2. **Mask Classification**:
   - A TensorFlow model (`saved_model.h5`) predicts whether a face has a mask (0: Mask, 1: No Mask).
   - Bounding boxes are drawn around detected faces with classification labels:
     - Green box: Mask detected.
     - Red box: No Mask detected.

3. **Frame Display**:
   - Processed frames are displayed with annotated results using `cv2_imshow`.

4. **Control**:
   - The program runs until the video ends or the user presses the `q` key.

---

## Requirements
Ensure the following dependencies are installed:
- Python 3.x
- TensorFlow
- Keras
- OpenCV
- NumPy

Install dependencies using:
```bash
pip install tensorflow keras opencv-python-headless numpy
```

---

## Setup
1. Download the required files:
   - `Face Mask detection using OpenCV.py` (main script).
   - `saved_model.h5` (pre-trained TensorFlow model).
   - `haarcascade_frontalface_default.xml` (Haar Cascade classifier).
   - `video.mp4` (input video).

2. Ensure all files are in the same directory.

---

## Usage
Run the following command to execute the script:
```bash
python Face Mask detection using OpenCV.py
```

### Script Execution
- The script processes the input video frame by frame.
- Detected faces are saved temporarily as images for classification.
- Press `q` to stop processing manually.

---

## Key Functions in the Code
- **Face Detection**:
  ```python
  faces = face_cascade.detectMultiScale(gray_img, 1.1, 6)
  ```
  Detects faces in the current frame.

- **Resizing Frames**:
  ```python
  dim = (int(color_img.shape[1]*scale/100), int(color_img.shape[0]*scale/100))
  color_img = cv2.resize(color_img, dim, interpolation=cv2.INTER_AREA)
  ```
  Scales the input frames for processing.

- **Mask Prediction**:
  ```python
  prediction = model.predict(img)
  ```
  Predicts the mask status of the extracted face image.

- **Drawing Results**:
  ```python
  cv2.rectangle(color_img, (x, y), (x+w, y+h), color, 3)
  cv2.putText(color_img, class_label, org, font, fontscale, color, thickness)
  ```
