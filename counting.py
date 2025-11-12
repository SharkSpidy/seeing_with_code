import cv2
from cvzone.HandTrackingModule import HandDetector

# Initialize camera and detector
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)  # Detect hands + draw landmarks

    if hands:
        hand = hands[0]
        fingers = detector.fingersUp(hand)  # [1,1,0,0,0] -> 2 fingers up
        # print(fingers)
        totalFingers = fingers.count(1)

        cv2.putText(img, f'Fingers: {totalFingers}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (255, 0, 255), 3)

    cv2.imshow("Finger Counter", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
