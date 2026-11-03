import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# ---------------- PATHS ----------------

BASE_DIR = Path(__file__).resolve().parent

YUNET_MODEL = BASE_DIR / "models" / "face_detection_yunet_2026may.onnx"
SFACE_MODEL = BASE_DIR / "models" / "face_recognition_sface_2021dec.onnx"
DATABASE_FILE = BASE_DIR / "data" / "registered_faces.json"

# ---------------- SETTINGS ----------------

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

POSES = [
    "Front - Look directly at the camera",
    "Left - Slowly turn your face left",
    "Right - Slowly turn your face right",
    "Up - Slightly raise your head",
    "Down - Slightly lower your head",
]

# ---------------- LOAD DATABASE ----------------

with open(DATABASE_FILE, "r", encoding="utf-8") as file:
    database = json.load(file)

person_name = input("Enter the existing registered name: ").strip()

person = next((p for p in database if p["name"].lower() == person_name.lower()), None)

if person is None:
    print("Person not found in database. No changes made.")
    raise SystemExit

# Keep the original embedding and add new embeddings separately.
if "embeddings" not in person:
    person["embeddings"] = [person["embedding"]]

print("Existing samples:", person.get("samples", 0))
print("New registration will ADD samples.")

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

# ---------------- START CAMERA ----------------

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam!")

camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

print("\nRegistration started.")
print("Follow the pose instructions shown on screen.")
print("Press SPACE to capture each pose.")
print("Press Q to cancel.")

# ---------------- CAPTURE POSES ----------------

new_embeddings = []

for pose_number, instruction in enumerate(POSES, start=1):

    captured = False

    while not captured:

        success, frame = camera.read()

        if not success:
            print("Could not read webcam frame.")
            break

        height, width = frame.shape[:2]
        detector.setInputSize((width, height))

        _, faces = detector.detect(frame)

        face_count = 0 if faces is None else len(faces)

        cv2.putText(
            frame,
            f"Pose {pose_number}/5",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            instruction,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )

        if face_count == 1:

            face = faces[0]
            x, y, w, h = face[:4].astype(int)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(
                frame,
                "One face detected - Press SPACE",
                (20, height - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        elif face_count == 0:

            cv2.putText(
                frame,
                "No face detected",
                (20, height - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        else:

            cv2.putText(
                frame,
                "Only one person should be in frame",
                (20, height - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Guided Face Registration", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Registration cancelled. No new samples saved.")
            camera.release()
            cv2.destroyAllWindows()
            raise SystemExit

        if key == 32 and face_count == 1:

            aligned_face = recognizer.alignCrop(frame, faces[0])
            embedding = recognizer.feature(aligned_face)

            embedding = embedding.flatten()
            embedding = embedding / np.linalg.norm(embedding)

            new_embeddings.append(embedding.tolist())

            print(f"Captured pose {pose_number}/5: {instruction.split(' - ')[0]}")
            captured = True

    if not success:
        break

camera.release()
cv2.destroyAllWindows()

# ---------------- SAVE NEW SAMPLES ----------------

if len(new_embeddings) == len(POSES):

    person["embeddings"].extend(new_embeddings)
    person["samples"] = len(person["embeddings"])
    person["updated_at"] = datetime.now().isoformat(timespec="seconds")

    with open(DATABASE_FILE, "w", encoding="utf-8") as file:
        json.dump(database, file, indent=4)

    print("\nRegistration completed successfully!")
    print("Person:", person["name"])
    print("New samples added:", len(new_embeddings))
    print("Total stored embeddings:", len(person["embeddings"]))

else:
    print("\nRegistration incomplete. No new samples were saved.")
