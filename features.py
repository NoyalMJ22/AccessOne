import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load face landmark model
base_options = python.BaseOptions(model_asset_path="face_landmarker.task")

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1
)

face_mesh = vision.FaceLandmarker.create_from_options(options)

prev_time = time.time()

def get_features(frame):
    global prev_time

    h, w, _ = frame.shape
    dwell_time = 0
    blink = 0
    eye_dist = 0

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    results = face_mesh.detect(mp_image)

    if results.face_landmarks:
        for face in results.face_landmarks:

            left_eye = face[33]
            right_eye = face[263]

            lx, ly = int(left_eye.x * w), int(left_eye.y * h)
            rx, ry = int(right_eye.x * w), int(right_eye.y * h)

            eye_dist = abs(lx - rx)

            current_time = time.time()
            dwell_time = current_time - prev_time

            if eye_dist < 15:
                blink = 1

            prev_time = current_time

    return eye_dist, dwell_time, blink