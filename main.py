import cv2
from cvzone.HandTrackingModule import HandDetector
from pynput.keyboard import Controller
from time import sleep
import cvzone

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)
keyboard = Controller()

# Keyboard layout
keys = [
    ["Q","W","E","R","T","Y","U","I","O","P"],
    ["A","S","D","F","G","H","J","K","L",";"],
    ["Z","X","C","V","B","N","M",",",".","/"]
]

# Button class
class Button:
    def __init__(self, pos, text, size=[85,85]):
        self.pos = pos
        self.size = size
        self.text = text

# Create button objects
buttonList = []
for i, row in enumerate(keys):
    for j, key in enumerate(row):
        buttonList.append(Button([100*j + 50, 100*i + 50], key))

# Function to draw buttons
def drawAll(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cvzone.cornerRect(img, (x, y, w, h), 20, rt=0)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), cv2.FILLED)
        cv2.putText(img, button.text, (x + 20, y + 65),
                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    return img

finalText = ""

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Detect hands
    hands, img = detector.findHands(img)  # Returns landmarks automatically
    lmList = []
    if hands:
        lmList = hands[0]['lmList']

    # Draw keyboard buttons
    img = drawAll(img, buttonList)

    if lmList:
        for button in buttonList:
            x, y = button.pos
            w, h = button.size

            if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
                cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (175, 0, 175), cv2.FILLED)
                cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

                # Get landmark points
                pointIndex = lmList[8]
                pointMiddle = lmList[12]

                # Distance
                l, _, _ = detector.findDistance(
                    (lmList[8][0], lmList[8][1]),
                    (lmList[12][0], lmList[12][1]),
                    img
                )
                if l < 30:
                    keyboard.press(button.text)
                    finalText += button.text
                    sleep(0.2)

    # Display typed text
    cv2.rectangle(img, (50, 350), (1200, 450), (175,0,175), cv2.FILLED)
    cv2.putText(img, finalText, (60, 430),
                cv2.FONT_HERSHEY_PLAIN, 5, (255,255,255), 5)

    cv2.imshow("Virtual Keyboard", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

