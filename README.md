# Face Recognition using Eigenfaces (PCA + Orthogonal Projection)

A simple face recognition system built from scratch using Linear Algebra concepts —
Eigenvalue Decomposition and Orthogonal Projection — without using any deep learning libraries.

--

## Concepts Used

- Principal Component Analysis (PCA)
- Eigenvalue & Eigenvector Decomposition
- Orthogonal Projection
- Nearest Neighbor Classification (Euclidean Distance)

---

## Tech Stack

- Python 3.x
- OpenCV — image reading and face detection
- NumPy — all matrix and eigenvalue operations
- Scikit-learn — eigenvector normalization
- Matplotlib — eigenface visualization

---
## Project Structure
face_recognition_project/
│
├── dataset/
│   ├── person1/        ← 8-10 training images of person 1
│   ├── person2/        ← 8-10 training images of person 2
│   └── person3/        ← 8-10 training images of person 3
│
├── test_images/        ← images to test recognition on
│
└── face_recognition.py
---

## How It Works

1. Each face image is converted to grayscale and cropped using Haar Cascade face detector
2. Cropped faces are flattened into 4096-dimensional vectors (64×64)
3. Mean face is computed and subtracted from all vectors (centering)
4. Covariance matrix is formed and eigenvalue decomposition is performed
5. Top 10 eigenvectors (Eigenfaces) are kept — these represent the most important facial patterns
6. Every face (train + test) is projected onto the eigenface subspace (Orthogonal Projection)
7. Recognition is done by finding the nearest neighbor using Euclidean distance in projected space
8. If distance exceeds a threshold, the person is marked as Unknown

---

## Installation

```bash
pip install numpy opencv-python scikit-learn matplotlib
```

---

## Usage

1. Add training images to `dataset/person_name/` subfolders
2. Add test images to `test_images/`
3. Run:

```bash
python face_recognition.py
```
## Sample Output
📂 Loading dataset...
Loaded: gigi hadid (0)
Loaded: kendall jenner (1)
Loaded: pat cummins (2)
Total images loaded: 30
🔢 Computing Eigenfaces (Eigenvalue Decomposition)...
Eigenfaces computed: 10 components
📐 Projecting training faces onto Eigenface subspace (Orthogonal Projection)...
Projection shape: (30, 10)
🔍 Recognizing test images...
test1.png  →  Recognized as: gigi hadid  (distance: 1423.45)
test2.png  →  Unknown Person             (distance: 3102.67)

## Limitations

- Works best with front-facing, clear face images
- Accuracy depends on number and variety of training images (recommended: 8-10 per person)
- Pure linear algebra approach — no deep learning, intentionally kept simple for academic purposes

---

## Academic Context

This project was built as part of a Linear Algebra course to demonstrate real-world applications
of eigenvalue problems and orthogonal projections in image processing and pattern recognition.
