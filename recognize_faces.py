import cv2
import json
import numpy as np
from pathlib import Path

# ---------------- CONFIGURATION ----------------

BASE_DIR = Path(__file__).resolve().parent

YUNET_MODEL = BASE_DIR / "models" / "face_detection_yunet_2026may.onnx"
SFACE_MODEL = BASE_DIR / "models" / "face_recognition_sface_2021dec.onnx"
DATABASE_FILE = BASE_DIR / "data" / "registered_faces.json"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

SIMILARITY_THRESHOLD = 0.40

# ---------------- LOAD DATABASE ----------------

with open(DATABASE_FILE, "r", encoding="utf-8") as file:
    database = json.load(file)

registered_faces = {}

for person in database:

    embeddings = person.get("embeddings")

    # Support older database entries
    if not embeddings:
        embeddings = [person["embedding"]]

    registered_faces[person["name"]] = [
        np.array(embedding, dtype=np.float32).reshape(1, -1) for embedding in embeddings
    ]

if not registered_faces:
    raise RuntimeError("No registered faces found!")

print("Registered people:")

for name, embeddings in registered_faces.items():
    print(f"{name}: {len(embeddings)} embeddings")

# ---------------- LOAD MODELS ----------------

detector = cv2.FaceDetectorYN.create(
    str(YUNET_MODEL),
    "",
    (CAMERA_WIDTH, CAMERA_HEIGHT),
    0.85,
    0.3,
    5000,
    cv2.dnn.DNN_BACKEND_OPENCV,
    cv2.dnn.DNN_TARGET_CPU,
)

recognizer = cv2.FaceRecognizerSF.create(
    str(SFACE_MODEL), "", cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU
)

print("\nModels loaded successfully!")

# ---------------- START WEBCAM ----------------

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam!")

camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

print("Webcam started!")
print("Press Q to exit.")

# ---------------- RECOGNITION LOOP ----------------

while True:

    success, frame = camera.read()

    if not success:
        print("Could not read webcam frame.")
        break

    height, width = frame.shape[:2]
    detector.setInputSize((width, height))

    _, faces = detector.detect(frame)

    if faces is not None:

        for face in faces:

            aligned_face = recognizer.alignCrop(frame, face)
            current_embedding = recognizer.feature(aligned_face)

            best_name = "Unknown"
            best_score = -1.0

            # Compare against every saved embedding
            for name, saved_embeddings in registered_faces.items():

                person_best_score = -1.0

                for saved_embedding in saved_embeddings:

                    score = recognizer.match(
                        current_embedding,
                        saved_embedding,
                        cv2.FaceRecognizerSF_FR_COSINE,
                    )

                    person_best_score = max(person_best_score, score)

                if person_best_score > best_score:
                    best_score = person_best_score
                    best_name = name

            # Apply recognition threshold
            if best_score < SIMILARITY_THRESHOLD:
                best_name = "Unknown"

            x, y, w, h = face[:4].astype(int)

            color = (0, 0, 255) if best_name == "Unknown" else (0, 255, 0)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            cv2.putText(
                frame,
                best_name,
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

            cv2.putText(
                frame,
                f"Similarity: {best_score:.3f}",
                (x, y + h + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

    cv2.imshow("Face Recognition - Multi Embedding", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---------------- CLEANUP ----------------

camera.release()
cv2.destroyAllWindows()

print("Face recognition stopped.")
