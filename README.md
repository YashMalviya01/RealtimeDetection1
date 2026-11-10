# Real-Time Face Detection & Real/Fake Classification

A Python computer-vision project for detecting faces from a live webcam feed and classifying detected faces as **Real** or **Fake**. The repository also includes scripts for collecting face images, preparing datasets, training a YOLO model, and running detection.

> **Note:** The current implementation uses two different approaches in its scripts: `FaceDectector.py` and `DataCollection.py` use CVZone's face detector, while `main.py` runs a custom Ultralytics YOLO model for Real/Fake classification. The Real/Fake decision logic in `DataCollection.py` is based on image heuristics; it should not be treated as a validated liveness or anti-spoofing system.

## Project Overview

The project explores a real-time face detection and classification workflow:

1. Capture frames from a webcam.
2. Detect faces and obtain their bounding boxes.
3. Collect face images and corresponding annotation information.
4. Prepare a dataset for model training.
5. Train a YOLOv8 model using the dataset configuration.
6. Run inference on a live webcam feed and display the predicted class and confidence score.

## Features

- Live webcam-based face detection.
- Face bounding boxes and detection confidence display.
- Real/Fake class display in the YOLO inference script.
- Face image collection with configurable confidence and blur thresholds.
- Normalized bounding-box annotation output.
- Dataset preparation and YOLO model-training scripts.
- FPS display during YOLO inference.

## Tech Stack

- **Python**
- **OpenCV** – webcam capture and image processing
- **CVZone** – face detection utilities and on-screen annotations
- **Ultralytics YOLO** – model training and inference
- **NumPy** – image calculations

## Repository Structure

```text
RealtimeDetection1/
│
├── DataCollection.py     # Collects face images and annotation data
├── FaceDectector.py      # Webcam face detection using CVZone
├── Train.py              # Trains a YOLO model
├── splitData.py          # Dataset splitting/preparation
├── main.py               # Real/Fake inference using a custom YOLO model
├── yoloTest.py           # YOLO testing script
├── cuda.py               # CUDA-related test/utility script
├── textfiletest.py       # Text/annotation file utility
├── test.txt              # Annotation output/sample data
├── yolov5su.pt           # YOLO model weights
└── yolov8n.pt            # YOLOv8 nano model weights
```

The dataset folders and the custom model referenced by `main.py` may need to be created or supplied separately.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YashMalviya01/RealtimeDetection1.git
cd RealtimeDetection1
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install opencv-python cvzone numpy ultralytics mediapipe
```

If you use a different Python or CUDA configuration, install the PyTorch/Ultralytics versions compatible with your system. A webcam is required for the live-camera scripts.

## Usage

### Face detection

Run the CVZone-based face detector:

```bash
python FaceDectector.py
```

The script opens the webcam, draws a bounding box around detected faces, and displays the detection confidence. Press **`q`** to exit.

### Collect face images

```bash
python DataCollection.py
```

The collection script uses the webcam and saves images under:

```text
Datasets/Data Collect/
```

It also appends normalized annotation information to `test.txt`. Review the configuration values near the top of the script before collecting data, especially the output directory, confidence threshold, blur threshold, and class ID.

### Train the YOLO model

The training script expects a dataset YAML file at:

```text
Datasets/SplitData/data0ffline.yaml
```

After preparing the dataset and correcting the YAML path if necessary, run:

```bash
python Train.py
```

The current training script uses `yolov8n.pt` as its starting model and specifies 300 epochs. Adjust the training configuration based on your dataset and hardware.

### Run Real/Fake inference

`main.py` loads a custom trained model and processes webcam frames. Before running it:

1. Make sure the trained weights file is available.
2. Update the model path in `main.py` to a valid path on your machine.
3. Confirm that the model's class order matches `classNames = ["Fake", "Real"]`.

Then run:

```bash
python main.py
```

The webcam window displays the predicted class and confidence score. Press **`q`** to exit.

## Configuration

Several settings can be adjusted directly in the scripts:

| Setting | Location | Purpose |
|---|---|---|
| Camera index | `cv2.VideoCapture(0)` | Selects the webcam device |
| Camera resolution | `DataCollection.py`, `main.py` | Sets the capture dimensions |
| Detection confidence | `DataCollection.py`, `FaceDectector.py` | Filters low-confidence face detections |
| Blur threshold | `DataCollection.py` | Controls the image sharpness filter |
| Class names | `main.py` | Maps model class IDs to `Fake` and `Real` |
| Model weights path | `main.py` | Points to the trained YOLO weights |
| Training epochs | `Train.py` | Sets the number of training epochs |

## Important Implementation Notes

- **Model path:** `main.py` currently contains a machine-specific absolute path to `n_version1.pt`. Replace it with a relative or local path that exists in your environment. That custom file is not listed among the repository's tracked files.
- **Dataset:** The training script expects a dataset YAML file and corresponding images/labels. Ensure the directory structure and YAML paths are correct before training.
- **Class mapping:** The order of `classNames` must exactly match the class IDs used when training the model.
- **Heuristic classification:** `DataCollection.py` labels a face using face area, aspect ratio, and brightness thresholds. These simple image heuristics can be affected by lighting, camera distance, pose, and device quality. They do not establish reliable spoof detection.
- **Privacy:** Collect and process face images only with appropriate consent, and store biometric data securely.

## Future Improvements

- Replace machine-specific file paths with configurable command-line arguments or a configuration file.
- Add a dependency file such as `requirements.txt`.
- Document the dataset format and provide a sample dataset YAML.
- Add evaluation metrics such as precision, recall, F1-score, and a confusion matrix.
- Test the classifier across varied lighting, camera quality, distances, and presentation-attack scenarios.
- Add structured logging and graceful handling for unavailable cameras or missing model files.

## Author

**Yash Malviya**

- GitHub: [@YashMalviya01](https://github.com/YashMalviya01)
- Repository: [RealtimeDetection1](https://github.com/YashMalviya01/RealtimeDetection1)

---

If you find this project useful, consider giving the repository a star.
