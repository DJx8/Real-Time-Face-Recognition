import cv2
import json
import numpy as np
import customtkinter as ctk

from PIL import Image
from pathlib import Path
from datetime import datetime

# ---------------- CONFIGURATION ----------------

BASE_DIR = Path(__file__).resolve().parent

YUNET_MODEL = BASE_DIR / "models" / "face_detection_yunet_2026may.onnx"
SFACE_MODEL = BASE_DIR / "models" / "face_recognition_sface_2021dec.onnx"
DATABASE_FILE = BASE_DIR / "data" / "registered_faces.json"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

SIMILARITY_THRESHOLD = 0.40

POSES = [
    "Front - Look directly at the camera",
    "Left - Slowly turn your face left",
    "Right - Slowly turn your face right",
    "Up - Slightly raise your head",
    "Down - Slightly lower your head",
]

# ---------------- GLOBAL VARIABLES ----------------

camera = None
camera_running = False
camera_image = None

detector = None
recognizer = None
registered_faces = {}

registration_active = False
registration_name = ""
registration_embeddings = []
registration_pose_index = 0

last_frame = None
last_faces = None

# ---------------- APPEARANCE ----------------

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ---------------- MAIN WINDOW ----------------

app = ctk.CTk()
app.title("Face Recognition Studio")
app.geometry("1100x700")
app.minsize(900, 600)

app.grid_columnconfigure(1, weight=1)
app.grid_rowconfigure(0, weight=1)

# ---------------- SIDEBAR ----------------

sidebar = ctk.CTkFrame(app, width=260, corner_radius=0)
sidebar.grid(row=0, column=0, sticky="nsew")
sidebar.grid_propagate(False)

ctk.CTkLabel(sidebar, text="FACE RECOGNITION", font=("Segoe UI", 19, "bold")).pack(
    pady=(40, 12), padx=12
)

ctk.CTkLabel(
    sidebar, text="LOCAL AI SYSTEM", font=("Segoe UI", 11), text_color="gray"
).pack(pady=(0, 35))

ctk.CTkLabel(
    sidebar, text="MENU", font=("Segoe UI", 12, "bold"), text_color="gray"
).pack(anchor="w", padx=25, pady=(5, 10))

ctk.CTkButton(
    sidebar, text="Start Camera", height=40, command=lambda: start_camera()
).pack(fill="x", padx=20, pady=8)

ctk.CTkButton(
    sidebar,
    text="Stop Camera",
    height=40,
    fg_color="#8B3030",
    hover_color="#A83A3A",
    command=lambda: stop_camera(),
).pack(fill="x", padx=20, pady=8)

# ---------------- REGISTRATION CONTROLS ----------------

name_entry = ctk.CTkEntry(sidebar, placeholder_text="Enter person's name", height=40)
name_entry.pack(fill="x", padx=20, pady=(30, 8))

register_button = ctk.CTkButton(
    sidebar, text="Register Person", height=40, command=lambda: start_registration()
)
register_button.pack(fill="x", padx=20, pady=8)

capture_button = ctk.CTkButton(
    sidebar,
    text="Capture Pose",
    height=40,
    state="disabled",
    command=lambda: capture_pose(),
)
capture_button.pack(fill="x", padx=20, pady=8)

cancel_button = ctk.CTkButton(
    sidebar,
    text="Cancel Registration",
    height=40,
    fg_color="#8B3030",
    hover_color="#A83A3A",
    state="disabled",
    command=lambda: cancel_registration(),
)
cancel_button.pack(fill="x", padx=20, pady=8)

registration_status = ctk.CTkLabel(
    sidebar, text="Registration inactive", text_color="gray", wraplength=210
)
registration_status.pack(padx=15, pady=15)

# ---------------- MAIN CONTENT ----------------

main_area = ctk.CTkFrame(app, corner_radius=0)
main_area.grid(row=0, column=1, sticky="nsew", padx=30, pady=25)

main_area.grid_columnconfigure(0, weight=1)
main_area.grid_rowconfigure(1, weight=1)

header = ctk.CTkFrame(main_area, fg_color="transparent")
header.grid(row=0, column=0, sticky="ew", pady=(5, 20))

ctk.CTkLabel(header, text="Dashboard", font=("Segoe UI", 26, "bold")).pack(side="left")

status_label = ctk.CTkLabel(
    header, text="● CAMERA OFF", text_color="gray", font=("Segoe UI", 12, "bold")
)
status_label.pack(side="right", padx=10)

# ---------------- CAMERA PREVIEW ----------------

camera_frame = ctk.CTkFrame(main_area, corner_radius=15)
camera_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 15))

camera_frame.grid_rowconfigure(0, weight=1)
camera_frame.grid_columnconfigure(0, weight=1)

camera_preview = ctk.CTkLabel(
    camera_frame,
    text="📷\n\nCAMERA PREVIEW\n\nClick Start Camera to begin",
    font=("Segoe UI", 20),
    text_color="gray",
)
camera_preview.grid(row=0, column=0, sticky="nsew")

# ---------------- INFORMATION PANEL ----------------

info_frame = ctk.CTkFrame(main_area, corner_radius=15)
info_frame.grid(row=2, column=0, sticky="ew")

info_frame.grid_columnconfigure((0, 1, 2), weight=1)

registered_count_label = ctk.CTkLabel(
    info_frame, text="REGISTERED PEOPLE\n0", font=("Segoe UI", 16, "bold")
)
registered_count_label.grid(row=0, column=0, padx=15, pady=20)

ctk.CTkLabel(
    info_frame, text="RECOGNITION ENGINE\nSFace + YuNet", font=("Segoe UI", 16, "bold")
).grid(row=0, column=1, padx=15, pady=20)

ctk.CTkLabel(
    info_frame, text="PROCESSING MODE\nCPU", font=("Segoe UI", 16, "bold")
).grid(row=0, column=2, padx=15, pady=20)


# ---------------- LOAD MODELS AND DATABASE ----------------


def load_recognition_system():
    global detector, recognizer, registered_faces

    try:
        if not YUNET_MODEL.exists():
            raise FileNotFoundError(f"YuNet model not found: {YUNET_MODEL}")

        if not SFACE_MODEL.exists():
            raise FileNotFoundError(f"SFace model not found: {SFACE_MODEL}")

        if not DATABASE_FILE.exists():
            raise FileNotFoundError(f"Face database not found: {DATABASE_FILE}")

        with open(DATABASE_FILE, "r", encoding="utf-8") as file:
            database = json.load(file)

        registered_faces = {}

        for person in database:
            embeddings = person.get("embeddings")

            if not embeddings:
                embeddings = [person["embedding"]]

            valid_embeddings = [
                np.array(embedding, dtype=np.float32).reshape(1, -1)
                for embedding in embeddings
            ]

            if valid_embeddings:
                registered_faces[person["name"]] = valid_embeddings

        if not registered_faces:
            raise RuntimeError("No registered faces found in database!")

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

        registered_count_label.configure(
            text=f"REGISTERED PEOPLE\n{len(registered_faces)}"
        )

        print("Recognition system loaded!")
        print("Registered people:")

        for name, embeddings in registered_faces.items():
            print(f"{name}: {len(embeddings)} embeddings")

        return True

    except Exception as error:
        print("Error loading recognition system:", error)

        status_label.configure(text="● SYSTEM ERROR", text_color="#E74C3C")

        return False


# ---------------- FACE RECOGNITION ----------------


def recognize_face(frame, face):

    aligned_face = recognizer.alignCrop(frame, face)
    current_embedding = recognizer.feature(aligned_face)

    best_name = "Unknown"
    best_score = -1.0

    for name, saved_embeddings in registered_faces.items():

        person_best_score = -1.0

        for saved_embedding in saved_embeddings:

            score = recognizer.match(
                current_embedding, saved_embedding, cv2.FaceRecognizerSF_FR_COSINE
            )

            person_best_score = max(person_best_score, score)

        if person_best_score > best_score:
            best_score = person_best_score
            best_name = name

    if best_score < SIMILARITY_THRESHOLD:
        best_name = "Unknown"

    return best_name, best_score


# ---------------- REGISTRATION FUNCTIONS ----------------


def start_registration():
    global registration_active
    global registration_name
    global registration_embeddings
    global registration_pose_index

    if not camera_running:
        registration_status.configure(
            text="Start the camera first!", text_color="#E74C3C"
        )
        return

    name = name_entry.get().strip()

    if not name:
        registration_status.configure(text="Please enter a name.", text_color="#E74C3C")
        return

    registration_name = name
    registration_embeddings = []
    registration_pose_index = 0
    registration_active = True

    capture_button.configure(state="normal")
    cancel_button.configure(state="normal")
    register_button.configure(state="disabled")
    name_entry.configure(state="disabled")

    registration_status.configure(text=f"Pose 1/5: {POSES[0]}", text_color="#2ecc71")

    print("Registration started for:", registration_name)


def capture_pose():
    global registration_pose_index
    global registration_active
    global registration_embeddings

    if not registration_active:
        return

    if last_frame is None or last_faces is None:
        return

    if len(last_faces) != 1:
        registration_status.configure(
            text="Keep exactly one face in the camera.", text_color="#E74C3C"
        )
        return

    try:
        aligned_face = recognizer.alignCrop(last_frame, last_faces[0])

        embedding = recognizer.feature(aligned_face)
        embedding = embedding.flatten()

        norm = np.linalg.norm(embedding)

        if norm == 0:
            raise ValueError("Invalid face embedding.")

        embedding = embedding / norm

        registration_embeddings.append(embedding.tolist())

        print(
            f"Captured pose {registration_pose_index + 1}/5: "
            f"{POSES[registration_pose_index].split(' - ')[0]}"
        )

        registration_pose_index += 1

        if registration_pose_index == len(POSES):
            save_registration()

        else:
            registration_status.configure(
                text=(
                    f"Pose {registration_pose_index + 1}/5: "
                    f"{POSES[registration_pose_index]}"
                ),
                text_color="#2ecc71",
            )

    except Exception as error:
        print("Capture error:", error)

        registration_status.configure(
            text="Could not capture face. Try again.", text_color="#E74C3C"
        )


def save_registration():
    global registration_active
    global registration_embeddings
    global registered_faces

    try:
        with open(DATABASE_FILE, "r", encoding="utf-8") as file:
            database = json.load(file)

        person = next(
            (
                item
                for item in database
                if item["name"].lower() == registration_name.lower()
            ),
            None,
        )

        if person is not None:

            embeddings = person.get("embeddings")

            if not embeddings:
                embeddings = [person["embedding"]]

            embeddings.extend(registration_embeddings)

            person["embeddings"] = embeddings
            person["samples"] = len(embeddings)
            person["updated_at"] = datetime.now().isoformat(timespec="seconds")

            print("Added samples to existing person:", person["name"])

        else:

            person = {
                "name": registration_name,
                "embedding": registration_embeddings[0],
                "embeddings": registration_embeddings,
                "samples": len(registration_embeddings),
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }

            database.append(person)

            print("New person registered:", registration_name)

        with open(DATABASE_FILE, "w", encoding="utf-8") as file:
            json.dump(database, file, indent=4)

        registered_faces[person["name"]] = [
            np.array(item, dtype=np.float32).reshape(1, -1)
            for item in person["embeddings"]
        ]

        registered_count_label.configure(
            text=f"REGISTERED PEOPLE\n{len(registered_faces)}"
        )

        registration_status.configure(
            text=f"{person['name']} registered successfully!", text_color="#2ecc71"
        )

        print("Registration completed successfully!")
        print("Total embeddings:", len(person["embeddings"]))

    except Exception as error:
        print("Database save error:", error)

        registration_status.configure(
            text="Could not save registration.", text_color="#E74C3C"
        )

    finally:
        registration_active = False
        capture_button.configure(state="disabled")
        cancel_button.configure(state="disabled")
        register_button.configure(state="normal")
        name_entry.configure(state="normal")


def cancel_registration():
    global registration_active
    global registration_embeddings
    global registration_pose_index

    registration_active = False
    registration_embeddings = []
    registration_pose_index = 0

    capture_button.configure(state="disabled")
    cancel_button.configure(state="disabled")
    register_button.configure(state="normal")
    name_entry.configure(state="normal")

    registration_status.configure(
        text="Registration cancelled. Nothing saved.", text_color="gray"
    )

    print("Registration cancelled.")


# ---------------- START CAMERA ----------------


def start_camera():
    global camera, camera_running

    if camera_running:
        return

    if not load_recognition_system():
        return

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        camera.release()
        camera = None

        status_label.configure(text="● CAMERA ERROR", text_color="#E74C3C")

        print("Could not open webcam!")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    camera_running = True

    status_label.configure(text="● RECOGNITION ACTIVE", text_color="#2ecc71")

    print("Camera and recognition started!")

    update_camera()


# ---------------- UPDATE CAMERA ----------------


def update_camera():
    global camera_image
    global last_frame
    global last_faces

    if not camera_running or camera is None:
        return

    success, frame = camera.read()

    if success:

        frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))

        detector.setInputSize((CAMERA_WIDTH, CAMERA_HEIGHT))

        _, faces = detector.detect(frame)

        last_frame = frame.copy()
        last_faces = faces

        face_count = 0

        if faces is not None:

            face_count = len(faces)

            for face in faces:

                try:
                    name, score = recognize_face(frame, face)

                    x, y, w, h = face[:4].astype(int)

                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                    cv2.putText(
                        frame,
                        name,
                        (x, max(25, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2,
                    )

                    cv2.putText(
                        frame,
                        f"Similarity: {score:.3f}",
                        (x, min(CAMERA_HEIGHT - 10, y + h + 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )

                except Exception as error:
                    print("Face recognition error:", error)

        # Registration instructions on camera
        if registration_active:

            pose_text = (
                f"REGISTRATION: {registration_name} | "
                f"POSE {registration_pose_index + 1}/5"
            )

            cv2.putText(
                frame,
                pose_text,
                (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                POSES[registration_pose_index],
                (15, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2,
            )

        # Display total face count
        cv2.putText(
            frame,
            f"Faces detected: {face_count}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        status_label.configure(
            text=f"● RECOGNITION ACTIVE | FACES: {face_count}", text_color="#2ecc71"
        )

        # Convert frame for CustomTkinter
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(frame)

        camera_image = ctk.CTkImage(
            light_image=pil_image, dark_image=pil_image, size=(640, 480)
        )

        camera_preview.configure(image=camera_image, text="")

    app.after(15, update_camera)


# ---------------- STOP CAMERA ----------------


def stop_camera():
    global camera
    global camera_running
    global camera_image
    global registration_active
    global registration_embeddings
    global registration_pose_index

    camera_running = False

    if camera is not None:
        camera.release()
        camera = None

    camera_image = None
    last_frame_reset = None

    if registration_active:
        registration_active = False
        registration_embeddings = []
        registration_pose_index = 0

        capture_button.configure(state="disabled")
        cancel_button.configure(state="disabled")
        register_button.configure(state="normal")
        name_entry.configure(state="normal")

        registration_status.configure(
            text="Registration stopped. Nothing saved.", text_color="gray"
        )

    camera_preview.configure(
        image=None, text="📷\n\nCAMERA PREVIEW\n\nClick Start Camera to begin"
    )

    status_label.configure(text="● CAMERA OFF", text_color="gray")

    print("Camera stopped!")


# ---------------- CLOSE APPLICATION ----------------


def close_application():
    stop_camera()
    app.destroy()


# ---------------- RUN APPLICATION ----------------

app.protocol("WM_DELETE_WINDOW", close_application)

print("Face Recognition Studio started!")

app.mainloop()
