from ultralytics import YOLO
import cv2
import time
import matplotlib.pyplot as plt
from openpyxl import Workbook
from datetime import datetime
import pygame
import os
import mediapipe as mp
from deepface import DeepFace

# INITIALIZE PYGAME
pygame.mixer.init()

# LOAD YOLO MODEL
model = YOLO("yolov8n.pt")

# START CAMERA
cap = cv2.VideoCapture(0)

# FPS VARIABLE
prev_time = 0

# TIMER VARIABLES
idle_start = None

# TIME TRACKING VARIABLES
working_seconds = 0
idle_seconds = 0
phone_seconds = 0

# SLEEP VARIABLES
sleep_counter = 0

# EMOTION VARIABLE
emotion = "Unknown"

# CREATE SCREENSHOT FOLDER
if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

# CREATE EXCEL FILE
workbook = Workbook()

sheet = workbook.active

sheet.title = "Employee Report"

sheet.append([
    "Time",
    "Status",
    "Working Time",
    "Idle Time",
    "Phone Usage Time",
    "Productivity",
    "Emotion"
])

# VIDEO RECORDING
fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter(
    'output.avi',
    fourcc,
    20.0,
    (640, 480)
)

# MEDIAPIPE FACE MESH
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True
)

# MAIN LOOP
while True:

    ret, frame = cap.read()

    if not ret:
        break

    # MIRROR EFFECT
    frame = cv2.flip(frame, 1)

    # RGB FRAME
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # FACE LANDMARKS
    results_mesh = face_mesh.process(
        rgb_frame
    )

    # YOLO DETECTION
    results = model(frame)

    person_detected = False
    phone_detected = False
    person_count = 0

    for r in results:

        boxes = r.boxes

        for box in boxes:

            cls = int(box.cls[0])

            label = model.names[cls]

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # DRAW RECTANGLE
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # SHOW LABEL
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # PERSON DETECTION
            if label == "person":

                person_detected = True

                person_count += 1

            # PHONE DETECTION
            if label == "cell phone":

                phone_detected = True

    # STATUS CHECK
    if person_detected:

        if idle_start is None:
            idle_start = time.time()

        idle_time = int(
            time.time() - idle_start
        )

        # IDLE CHECK
        if idle_time > 10:

            status = "Doing Nothing"

            idle_seconds += 1

        else:

            status = "Working"

            working_seconds += 1

    else:

        idle_start = None

        status = "Employee Missing"

    # PHONE DETECTION
    if phone_detected:

        status = "Using Mobile Phone"

        phone_seconds += 1

        # SAVE SCREENSHOT
        filename = datetime.now().strftime(
            "screenshots/%Y%m%d_%H%M%S.jpg"
        )

        cv2.imwrite(filename, frame)

        # PLAY SOUND
        pygame.mixer.music.load(
            "alarm.wav"
        )

        pygame.mixer.music.play()

    # SLEEP DETECTION
    if results_mesh.multi_face_landmarks:

        for face_landmarks in results_mesh.multi_face_landmarks:

            left_eye = face_landmarks.landmark[145]

            right_eye = face_landmarks.landmark[374]

            eye_distance = abs(
                left_eye.y - right_eye.y
            )

            # EYES CLOSED
            if eye_distance < 0.01:

                sleep_counter += 1

            else:

                sleep_counter = 0

            # SLEEP STATUS
            if sleep_counter > 20:

                status = "Sleeping"

                cv2.putText(
                    frame,
                    "SLEEPING DETECTED!",
                    (20, 400),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

    # EMOTION DETECTION
    try:

        analysis = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False
        )

        emotion = analysis[0][
            'dominant_emotion'
        ]

    except:

        emotion = "Unknown"

    # CONVERT TIME TO MINUTES
    working_minutes = round(
        working_seconds / 60,
        2
    )

    idle_minutes = round(
        idle_seconds / 60,
        2
    )

    phone_minutes = round(
        phone_seconds / 60,
        2
    )

    # PRODUCTIVITY SCORE
    total = (
        working_seconds +
        idle_seconds +
        phone_seconds
    )

    if total > 0:

        productivity = int(
            (working_seconds / total) * 100
        )

    else:

        productivity = 0

    # SAVE STATUS TO EXCEL
    current_time = datetime.now().strftime(
        "%H:%M:%S"
    )

    sheet.append([
        current_time,
        status,
        working_minutes,
        idle_minutes,
        phone_minutes,
        productivity,
        emotion
    ])

    # SHOW STATUS
    cv2.putText(
        frame,
        f"Status: {status}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )

    # WORKING TIME
    cv2.putText(
        frame,
        f"Working: {working_minutes} min",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # IDLE TIME
    cv2.putText(
        frame,
        f"Doing Nothing: {idle_minutes} min",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # PHONE TIME
    cv2.putText(
        frame,
        f"Phone Usage: {phone_minutes} min",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    # PRODUCTIVITY
    cv2.putText(
        frame,
        f"Productivity: {productivity}%",
        (20, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # FPS COUNTER
    current_time_fps = time.time()

    fps = 1 / (
        current_time_fps - prev_time
    )

    prev_time = current_time_fps

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    # WARNING MESSAGE
    if phone_minutes > 1:

        cv2.putText(
            frame,
            "WARNING: Excess Mobile Usage!",
            (20, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3
        )

    # EMPLOYEE COUNT
    cv2.putText(
        frame,
        f"Employees: {person_count}",
        (20, 320),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 255),
        2
    )

    # LIVE CLOCK
    live_time = datetime.now().strftime(
        "%d-%m-%Y %H:%M:%S"
    )

    cv2.putText(
        frame,
        live_time,
        (20, 360),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # SHOW EMOTION
    cv2.putText(
        frame,
        f"Emotion: {emotion}",
        (20, 440),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # LIVE GRAPH
    graph_x = 450
    graph_y = 50
    bar_width = 40

    # GRAPH BACKGROUND
    cv2.rectangle(
        frame,
        (430, 20),
        (620, 220),
        (50, 50, 50),
        -1
    )

    # WORKING BAR
    cv2.rectangle(
        frame,
        (graph_x, 200),
        (
            graph_x + bar_width,
            200 - int(working_minutes * 10)
        ),
        (0, 255, 0),
        -1
    )

    # IDLE BAR
    cv2.rectangle(
        frame,
        (graph_x + 60, 200),
        (
            graph_x + 60 + bar_width,
            200 - int(idle_minutes * 10)
        ),
        (0, 255, 255),
        -1
    )

    # PHONE BAR
    cv2.rectangle(
        frame,
        (graph_x + 120, 200),
        (
            graph_x + 120 + bar_width,
            200 - int(phone_minutes * 10)
        ),
        (255, 0, 0),
        -1
    )

    # LABELS
    cv2.putText(
        frame,
        "W",
        (graph_x, 215),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        "I",
        (graph_x + 60, 215),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        "P",
        (graph_x + 120, 215),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        2
    )

    # SAVE VIDEO
    out.write(frame)

    # SHOW WINDOW
    cv2.imshow(
        "Employee Monitoring System",
        frame
    )

    # PRESS Q TO EXIT
    if cv2.waitKey(1) == ord('q'):
        break

# CLOSE CAMERA
cap.release()

out.release()

cv2.destroyAllWindows()

# SAVE EXCEL FILE
workbook.save(
    "employee_report.xlsx"
)

# FINAL REPORT
print("\n===== FINAL REPORT =====")

print(
    f"Working Time: {working_minutes} min"
)

print(
    f"Idle Time: {idle_minutes} min"
)

print(
    f"Phone Usage: {phone_minutes} min"
)

print(
    f"Productivity: {productivity:.2f}%"
)

print(
    f"Emotion: {emotion}"
)

# PRODUCTIVITY CHART
labels = [
    "Working",
    "Doing Nothing",
    "Phone Usage"
]

values = [
    working_seconds,
    idle_seconds,
    phone_seconds
]

plt.bar(labels, values)

plt.title(
    "Employee Productivity Analysis"
)

plt.xlabel("Activity")

plt.ylabel("Time (seconds)")

plt.show()