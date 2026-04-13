import os
import cv2
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt


# Add this at the top with other imports
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def crop_face(img):
    """Detect and crop face from image. Returns None if no face found."""
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    # Take the largest detected face
    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    face = img[y:y+h, x:x+w]
    return cv2.resize(face, IMG_SIZE)
# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_PATH = "dataset"
TEST_PATH = "test_images"
IMG_SIZE = (64, 64)       # all faces resized to this
NUM_COMPONENTS = 10        # number of eigenfaces to keep

# ─────────────────────────────────────────────
# STEP 1: LOAD IMAGES
# ─────────────────────────────────────────────
def load_dataset(dataset_path):
    faces = []
    labels = []
    label_names = {}
    label_id = 0

    for person_name in sorted(os.listdir(dataset_path)):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue

        label_names[label_id] = person_name
        for img_file in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = crop_face(img)
            if img is None:
                print(f"    No face found in {img_file}, skipping")
                continue
            faces.append(img.flatten().astype(np.float64))  # flatten to 1D vector
            labels.append(label_id)

        print(f"  Loaded: {person_name} ({label_id})")
        label_id += 1

    return np.array(faces), np.array(labels), label_names


# ─────────────────────────────────────────────
# STEP 2: PCA / EIGENFACES
# ─────────────────────────────────────────────
def compute_eigenfaces(faces, num_components):
    # Mean face
    mean_face = np.mean(faces, axis=0)
    centered = faces - mean_face  # subtract mean from each face

    # Covariance matrix (using trick: A^T A instead of A A^T for efficiency)
    A = centered  # shape: (num_images, pixels)
    cov_matrix = np.dot(A, A.T)  # (num_images x num_images) — eigenvalue problem here

    # Eigenvalue decomposition — CORE of the project
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort by descending eigenvalue
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_idx]

    # Convert to image-space eigenvectors (eigenfaces)
    eigenfaces = np.dot(A.T, eigenvectors).T  # shape: (num_images, pixels)

    # Normalize eigenfaces
    eigenfaces = normalize(eigenfaces)

    # Keep only top components
    eigenfaces = eigenfaces[:num_components]

    return mean_face, eigenfaces


# ─────────────────────────────────────────────
# STEP 3: PROJECT FACES ONTO EIGENFACE SPACE
# (Orthogonal Projection)
# ─────────────────────────────────────────────
def project(faces, mean_face, eigenfaces):
    centered = faces - mean_face
    # W = A @ eigenfaces.T  → coordinates in eigenface subspace
    return np.dot(centered, eigenfaces.T)


# ─────────────────────────────────────────────
# STEP 4: RECOGNIZE A FACE
# (Nearest Neighbor in projected space)
# ─────────────────────────────────────────────
def recognize(test_face_vector, train_projections, labels, label_names):
    test_projection = project(test_face_vector.reshape(1, -1), mean_face, eigenfaces)
    
    # Euclidean distance to all training projections
    distances = np.linalg.norm(train_projections - test_projection, axis=1)
    
    best_match_idx = np.argmin(distances)
    best_label = labels[best_match_idx]
    best_distance = distances[best_match_idx]

    return label_names[best_label], best_distance


# ─────────────────────────────────────────────
# STEP 5: VISUALIZE EIGENFACES
# ─────────────────────────────────────────────
def show_eigenfaces(eigenfaces, num_show=5):
    fig, axes = plt.subplots(1, num_show, figsize=(15, 3))
    fig.suptitle("Top Eigenfaces (Principal Components)", fontsize=13)
    for i in range(num_show):
        face_img = eigenfaces[i].reshape(IMG_SIZE)
        face_img = (face_img - face_img.min()) / (face_img.max() - face_img.min())  # normalize for display
        axes[i].imshow(face_img, cmap='gray')
        axes[i].set_title(f"Eigenface {i+1}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
print("\n📂 Loading dataset...")
faces, labels, label_names = load_dataset(DATASET_PATH)
print(f"  Total images loaded: {len(faces)}")
print(f"  People: {list(label_names.values())}")

print("\n🔢 Computing Eigenfaces (Eigenvalue Decomposition)...")
mean_face, eigenfaces = compute_eigenfaces(faces, NUM_COMPONENTS)
print(f"  Eigenfaces computed: {eigenfaces.shape[0]} components")

print("\n📐 Projecting training faces onto Eigenface subspace (Orthogonal Projection)...")
train_projections = project(faces, mean_face, eigenfaces)
print(f"  Projection shape: {train_projections.shape}")

print("\n🖼️  Showing Eigenfaces...")
show_eigenfaces(eigenfaces, num_show=min(5, NUM_COMPONENTS))

# ─── TEST ────────────────────────────────────
print("\n🔍 Recognizing test images...")
for img_file in os.listdir(TEST_PATH):
    img_path = os.path.join(TEST_PATH, img_file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  Could not read: {img_file}")
        continue

    img_cropped = crop_face(img)
    if img_cropped is None:
        print(f"  No face detected in {img_file}, skipping")
        continue
    img_resized = img_cropped.flatten().astype(np.float64)
    name, distance = recognize(img_resized, train_projections, labels, label_names)

    print(f"  {img_file}  →  Recognized as: {name}  (distance: {distance:.2f})")


test_proj = project(img_resized.reshape(1, -1), mean_face, eigenfaces)
for pid, pname in label_names.items():
    mask = labels == pid
    avg_proj = train_projections[mask].mean(axis=0)
    dist = np.linalg.norm(avg_proj - test_proj)
    print(f"    Distance to {pname}: {dist:.2f}")

print("\n✅ Done!")