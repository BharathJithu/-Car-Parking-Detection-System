import cv2
import cvzone
import numpy as np
import firebase_admin
from firebase_admin import credentials, db

# Initialize Firebase
cred = credentials.Certificate("credentials.json")  # Use the downloaded JSON credentials
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://parking-4a13b-default-rtdb.firebaseio.com/'
})

# Define the positions of the parking slots based on the image layout
positions = [
    (20, 100), (120, 100), (220, 100), (320, 100),
    (20, 250), (120, 250), (220, 250), (320, 250)
]

width, height = 70, 130

# Setup Video Capture with lower resolution for faster processing
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower the resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load and resize logo once
logo = cv2.imread('logo.png')
size = 100
logo = cv2.resize(logo, (size, size))

# Create a mask of the logo once
img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)

# Firebase update function with state change optimization
prev_space_status = ['free'] * len(positions)

def update_firebase(space_status):
    free_slots = space_status.count('free')
    filled_slots = len(positions) - free_slots
    # Update Firebase only if there are changes
    db.reference('parking_status').set({
        'free_slots': free_slots,
        'filled_slots': filled_slots
    })

    for i, status in enumerate(space_status):
        # Update each parking slot only when its status changes
        if prev_space_status[i] != status:
            db.reference(f'parking_slots/slot_{i+1}').set(status)
            prev_space_status[i] = status  # Update previous status

def checkParkingSpace(imgPro):
    space_status = []
    for i, pos in enumerate(positions):
        x, y = pos
        imgCrop = imgPro[y:y + height, x:x + width]
        count = cv2.countNonZero(imgCrop)

        if count < 750:
            color = (0, 255, 0)
            text = 'Free'
            space_status.append('free')
        else:
            color = (0, 0, 255)
            text = 'Filled'
            space_status.append('filled')

        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, 2)
        cvzone.putTextRect(img, text, (x, y + height - 3), scale=1, thickness=1, offset=0, colorR=color)

    # Update Firebase only when there is a change in slot status
    update_firebase(space_status)

    free_slots = space_status.count('free')
    filled_slots = len(positions) - free_slots
    cvzone.putTextRect(img, 'Inker AI Parking Slot Detection System', (50, 50), scale=1.5, thickness=1, offset=10, colorR=(227, 204, 148))
    cvzone.putTextRect(img, f'Free: {free_slots}', (50, 100), scale=1, thickness=1, offset=10, colorR=(94, 10, 6))
    cvzone.putTextRect(img, f'Filled: {filled_slots}', (150, 100), scale=1, thickness=1, offset=10, colorR=(25, 101, 224))

while True:
    success, img = cap.read()
    if not success:
        break  # Break if no frames are captured

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

    # Check parking space
    checkParkingSpace(imgDilate)

    # Region of Image (ROI) where we want to insert logo
    roi = img[-size-10:-10, -size-10:-10]
    roi[np.where(mask)] = 0  # Zero out the area where the logo is placed
    roi += logo

    # Full screen
    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("window", img)

    # Press 'q' to exit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
