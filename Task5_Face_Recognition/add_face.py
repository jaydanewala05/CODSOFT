"""
add_face.py  –  Capture training images for a new person via webcam.
Usage:  python add_face.py
"""

import cv2
import os

KNOWN_FACES_DIR = "known_faces"
CASCADE_PATH    = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade    = cv2.CascadeClassifier(CASCADE_PATH)


def capture_faces(person_name: str, num_samples: int = 30):
    save_dir = os.path.join(KNOWN_FACES_DIR, person_name)
    os.makedirs(save_dir, exist_ok=True)

    cap   = cv2.VideoCapture(0)
    count = 0
    print(f"\n[INFO] Capturing {num_samples} face samples for '{person_name}'.")
    print("[INFO] Look at the camera. Press 'q' to stop early.\n")

    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        for (x, y, w, h) in rects:
            face_img = gray[y:y+h, x:x+w]
            out_path = os.path.join(save_dir, f"{person_name}_{count:03d}.jpg")
            cv2.imwrite(out_path, face_img)
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Saved {count}/{num_samples}", (x, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        cv2.putText(frame, f"Capturing: {person_name}  [{count}/{num_samples}]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.imshow("Add Face – CodSoft", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[OK] Saved {count} images to '{save_dir}'")
    print("[INFO] Now run face_app.py and choose option 3 to re-train the model.")


if __name__ == "__main__":
    name = input("Enter person's name (no spaces, e.g. John): ").strip()
    if name:
        capture_faces(name)
    else:
        print("[ERR] Name cannot be empty.")