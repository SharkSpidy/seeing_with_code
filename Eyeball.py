import cv2
import mediapipe as mp
import pyautogui

# Initialize
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

while True:
    ret, frame = cam.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    frame_h, frame_w, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Landmark IDs for right and left iris centers
            right_iris = face_landmarks.landmark[474:478]
            left_iris = face_landmarks.landmark[469:473]

            for idx, iris in enumerate(right_iris):
                x = int(iris.x * frame_w)
                y = int(iris.y * frame_h)
                cv2.circle(frame, (x, y), 2, (255, 0, 255), -1)

            # Use one iris (e.g., right) to move the cursor
            x = int(right_iris[0].x * frame_w)
            y = int(right_iris[0].y * frame_h)

            # Map camera coordinates to screen size
            screen_x = screen_w / frame_w * x
            screen_y = screen_h / frame_h * y
            pyautogui.moveTo(screen_x, screen_y)

            # Blink detection (using eyelid landmarks)
            left_eye = [face_landmarks.landmark[i] for i in [145, 159]]
            right_eye = [face_landmarks.landmark[i] for i in [374, 386]]

            # Measure blink distance
            left_blink = abs(left_eye[0].y - left_eye[1].y)
            right_blink = abs(right_eye[0].y - right_eye[1].y)

            if left_blink < 0.003 and right_blink < 0.003:
                pyautogui.click()
                pyautogui.sleep(0.5)

    cv2.imshow('Eye Controlled Cursor', frame)
    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
