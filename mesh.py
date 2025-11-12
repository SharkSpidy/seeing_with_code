import cv2
import mediapipe as mp
import numpy as np
import cvzone

mpFace = mp.solutions.face_mesh
faceMesh = mpFace.FaceMesh(max_num_faces=1)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

overlay = cv2.imread("glasses.png", cv2.IMREAD_UNCHANGED)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            h, w, _ = img.shape
            landmark_points = []
            for id, lm in enumerate(faceLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_points.append((cx, cy))

            left_eye = landmark_points[33]
            right_eye = landmark_points[263]
            width = int(((right_eye[0] - left_eye[0]) * 1.8))
            x = int(left_eye[0] - width / 4)
            y = int(left_eye[1] - width / 6)

            overlay_resized = cv2.resize(overlay, (width, int(width * 0.4)))

            img = cvzone.overlayPNG(img, overlay_resized, [x, y])

    cv2.imshow("Face Filter", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
