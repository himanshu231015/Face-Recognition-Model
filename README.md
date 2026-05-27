# Face Recognition & Matching System

A clean and interactive Face Recognition and Matching system built with Python, OpenCV, and dlib. This system lets you register faces (via webcam or manual image grouping) and match them in real-time.

---

## 📂 Project Structure

```text
Face-Recognition-Model/
├── dataset/               # Folder containing registered person folders
│   ├── John_Doe/          # Folder name is used as the person's name
│   │   ├── face_01.jpg
│   │   └── face_02.jpg
├── model.py               # Core logic (training, prediction, face cropping, IOU merge)
├── train.py               # CLI script to train the model from dataset folder
├── predict.py             # CLI script to match/identify faces in a static image file
├── register_face.py       # Interactive script to register a new face using your webcam
├── recognize_webcam.py    # Real-time face matcher that displays webcam stream with overlay
├── requirements.txt       # Python dependencies list
└── README.md              # Documentation
```

---

## 🛠️ Setup & Installation

### 1. Prerequisites
Make sure Python 3.8+ is installed on your computer.

### 2. Install dlib (Windows)
Installing `dlib` via standard `pip` can be difficult on Windows. It is highly recommended to install it using Conda:
```bash
conda install -c conda-forge dlib
```

### 3. Install Python Dependencies
Once `dlib` is installed, run:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Use

### Step 1: Register a Face
You can register a person's face easily using your webcam:
```bash
python register_face.py
```
1. It will ask for the person's name (e.g., `John Doe`).
2. The webcam will open. Fit your face inside the bounding guide.
3. Press `SPACE` to capture a frame. Capture **5 different photos** (with different head angles or expressions for higher accuracy).
4. Once completed, it will prompt you if you want to train the model immediately. Type `y` to train.

*(Alternatively, you can manually create a folder inside the `dataset/` directory named after the person, e.g., `dataset/Jane_Smith/` and place clear face images inside it.)*

### Step 2: Train the Model
If you didn't auto-train during registration, or manually added images, run:
```bash
python train.py
```
This extracts face encodings and compiles them into a fast KNN classifier (`face_model.pkl`). It also caches encodings (`encodings_cache.pkl`) to speed up future training sessions.

### Step 3: Run Real-time Face Matcher
To run face recognition in real-time on your webcam feed:
```bash
python recognize_webcam.py
```
- It will highlight detected faces in green with their registered names and confidence percentages.
- Unregistered faces will be highlighted in red as **"Unknown"**.
- Press `Q` to quit.

### Step 4: Run Static Image Prediction
To test face recognition on any static image file:
```bash
python predict.py path/to/your/image.jpg
```

---

## ⚙️ How it Works under the Hood
1. **Hybrid Face Detection**: Uses dlib's **HOG (Histogram of Oriented Gradients)** for fast, reliable frontal face detection, and falls back to **OpenCV Haar Cascades** for profile or tilted faces. Detections are merged using **Intersection Over Union (IOU)**.
2. **K-Nearest Neighbors (KNN) Classifier**: Converts faces to 128D encodings and classifies them using 1-NN mapping.
# Face-Recognition-Model
