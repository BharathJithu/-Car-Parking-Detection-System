import cv2
import pickle
import cvzone
import numpy as np
import firebase_admin
from firebase_admin import credentials, db

# Initialize Firebase
cred = credentials.Certificate("path_to_serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://your-database-url.firebaseio.com/'
})

# Video feed
cap = cv2.VideoCapture(0)

with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

width, height = 70, 130

# Logo setup
logo = cv2.imread('logo.png')
size = 100
logo = cv2.resize(logo, (size, size))

# Create a mask of the logo
img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)


# Firebase update function
def updateFirebase(free, filled):
    ref = db.reference('ParkingSlots')
    ref.update({
        'free': free,
        'filled': filled
    })


def checkParkingSpace(imgPro):
    spaceCounter = 0

    for pos in posList:
        x, y = pos
        imgCrop = imgPro[y:y + height, x:x + width]
        count = cv2.countNonZero(imgCrop)

        if count < 750:
            color = (0, 255, 0)
            thickness = 2
            spaceCounter += 1
            text = 'Free'
        else:
            color = (0, 0, 255)
            thickness = 2
            text = 'Filled'

        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        cvzone.putTextRect(img, str(text), (x, y + height - 3), scale=1, thickness=1, offset=0, colorR=color)

    # Update Free and Filled slots on the image
    cvzone.putTextRect(img, 'Free: ' + str(spaceCounter), (50, 100), scale=1, thickness=1, offset=10,
                       colorR=(94, 10, 6))
    filled = len(posList) - spaceCounter
    cvzone.putTextRect(img, 'Filled: ' + str(filled), (150, 100), scale=1, thickness=1, offset=10,
                       colorR=(25, 101, 224))

    # Update Firebase with the free and filled slots
    updateFirebase(free=spaceCounter, filled=filled)


while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

    checkParkingSpace(imgDilate)

    # Region of Image (ROI), where we want to insert logo
    roi = img[-size - 10:-10, -size - 10:-10]
    roi[np.where(mask)] = 0
    roi += logo

    # Full screen display
    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("window", img)

    cv2.waitKey(10)
