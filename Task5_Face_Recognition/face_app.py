"""
Task 5
Face Detection & Recognition
Supports: Images + Webcam
Author: jay danewala
"""

import cv2
import os
import numpy as np
import pickle
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
KNOWN_FACES_DIR = "known_faces"       # folder with subfolders per person
MODEL_FILE      = "face_model.pkl"    # saved trained model
OUTPUT_DIR      = "output_results"    # saved annotated images
CASCADE_PATH    = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# FACE DETECTOR  (Haar Cascade)
# ─────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def detect_faces(gray_img):
    """Return list of (x, y, w, h) bounding boxes."""
    return face_cascade.detectMultiScale(
        gray_img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )


# ─────────────────────────────────────────────
# FACE RECOGNIZER  (LBPH – works offline, no GPU needed)
# ─────────────────────────────────────────────
recognizer = cv2.face.LBPHFaceRecognizer_create()
label_map   = {}   # int → name


def train_model():
    """
    Scan known_faces/<PersonName>/*.jpg|png, train LBPH model, save to disk.
    Call this after adding new faces.
    """
    global label_map
    faces, labels = [], []
    label_map = {}
    idx = 0

    for person_name in sorted(os.listdir(KNOWN_FACES_DIR)):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        label_map[idx] = person_name
        for img_file in os.listdir(person_dir):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(person_dir, img_file)
            img  = cv2.imread(path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = detect_faces(gray)
            for (x, y, w, h) in rects:
                faces.append(gray[y:y+h, x:x+w])
                labels.append(idx)
        idx += 1

    if not faces:
        print("[WARN] No training faces found. Add images to known_faces/<Name>/")
        return False

    recognizer.train(faces, np.array(labels))
    recognizer.save("lbph_model.yml")
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(label_map, f)
    print(f"[OK] Model trained on {len(faces)} face(s) across {len(label_map)} person(s).")
    return True


def load_model():
    """Load previously trained model from disk."""
    global label_map
    if os.path.exists("lbph_model.yml") and os.path.exists(MODEL_FILE):
        recognizer.read("lbph_model.yml")
        with open(MODEL_FILE, "rb") as f:
            label_map = pickle.load(f)
        print(f"[OK] Model loaded — knows {len(label_map)} person(s): {list(label_map.values())}")
        return True
    print("[INFO] No saved model found. Run train_model() first.")
    return False


def recognize_face(face_roi_gray):
    """Return (name, confidence). Lower confidence = better match."""
    if not label_map:
        return "Unknown", 999
    label, confidence = recognizer.predict(face_roi_gray)
    name = label_map.get(label, "Unknown")
    # LBPH: confidence < 80 → good match
    if confidence > 80:
        name = "Unknown"
    return name, round(confidence, 1)


# ─────────────────────────────────────────────
# ANNOTATE FRAME
# ─────────────────────────────────────────────
def annotate_frame(frame, do_recognition=True):
    """Detect (and optionally recognize) faces; draw boxes + labels."""
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detect_faces(gray)

    for (x, y, w, h) in rects:
        face_roi = gray[y:y+h, x:x+w]

        if do_recognition and label_map:
            name, conf = recognize_face(face_roi)
            color = (0, 200, 0) if name != "Unknown" else (0, 60, 220)
            label_text = f"{name}  ({conf})"
        else:
            name, color = "Face", (255, 160, 0)
            label_text  = "Face Detected"

        # bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # label background
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x, y - th - 10), (x + tw + 6, y), color, -1)
        cv2.putText(frame, label_text, (x + 3, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # face count
    cv2.putText(frame, f"Faces: {len(rects)}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    return frame, len(rects)


# ─────────────────────────────────────────────
# MODE 1 – Process a single image
# ─────────────────────────────────────────────
def process_image(image_path, do_recognition=True):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERR] Cannot read image: {image_path}")
        return

    result, count = annotate_frame(img, do_recognition)
    print(f"[OK] Detected {count} face(s) in '{image_path}'")

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out  = os.path.join(OUTPUT_DIR, f"result_{ts}.jpg")
    cv2.imwrite(out, result)
    print(f"[OK] Saved result → {out}")

    cv2.imshow("Face Detection & Recognition", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# MODE 2 – Live webcam
# ─────────────────────────────────────────────
def run_webcam(do_recognition=True):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERR] Cannot open webcam.")
        return

    print("[INFO] Webcam started. Press 's' to save frame, 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result, _ = annotate_frame(frame, do_recognition)
        cv2.putText(result, "Press 's' save | 'q' quit", (10, result.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("Live Face Recognition – CodSoft", result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = os.path.join(OUTPUT_DIR, f"webcam_{ts}.jpg")
            cv2.imwrite(out, result)
            print(f"[OK] Frame saved → {out}")

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# MAIN MENU
# ─────────────────────────────────────────────
def main():
    print("\n" + "="*50)
    print("  CodSoft AI – Face Detection & Recognition")
    print("="*50)

    # Try to load existing model; train if not found
    model_ready = load_model()
    if not model_ready:
        print("\n[INFO] Training model from known_faces/ directory...")
        model_ready = train_model()

    print("\nOptions:")
    print("  1. Detect & Recognize in an IMAGE")
    print("  2. Live WEBCAM detection & recognition")
    print("  3. Re-train model (add new faces first)")
    print("  4. Quit")

    while True:
        choice = input("\nEnter choice (1-4): ").strip()

        if choice == "1":
            path = input("Enter image path: ").strip().strip('"')
            process_image(path, do_recognition=model_ready)

        elif choice == "2":
            run_webcam(do_recognition=model_ready)

        elif choice == "3":
            model_ready = train_model()

        elif choice == "4":
            print("Bye!")
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()