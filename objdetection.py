import cv2
import numpy as np
from random import randint


def nothing(x):
    pass


# Create a black image, a window
img = np.zeros((300, 512, 3), np.uint8)
img2 = np.zeros((300, 512, 3), np.uint8)
cv2.namedWindow('low')
cv2.namedWindow('high')

# create trackbars for color change
cv2.createTrackbar('r','low',0,255,nothing)
cv2.createTrackbar('g','low',0,255,nothing)
cv2.createTrackbar('b','low',0,255,nothing)
cv2.createTrackbar('R','high',0,255,nothing)
cv2.createTrackbar('G','high',0,255,nothing)
cv2.createTrackbar('B','high',0,255,nothing)

# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
switch2 = '0 : OFF \n1 : ON'

cv2.createTrackbar(switch, 'low',0,1,nothing)
cv2.createTrackbar(switch2, 'high',0,1,nothing)


cap = cv2.VideoCapture(0)
cap.set(3, 680)
cap.set(4, 480)
key = ord('a')
prev = (300, 200)
next = prev
high = np.array([120, 255, 255])
low = np.array([0, 30, 140])
wh = np.array([100, 255, 255])
wl = np.array([0, 125, 150])
firstFrame = None

while key != ord('q'):

    cv2.imshow('low', img)
    r = cv2.getTrackbarPos('r', 'low')
    g = cv2.getTrackbarPos('g', 'low')
    b = cv2.getTrackbarPos('b', 'low')
    s = cv2.getTrackbarPos(switch, 'low')

    if s == 0:
        img[:] = 0
    else:
        img[:] = [b, g, r]

    cv2.imshow('high', img2)
    rh = cv2.getTrackbarPos('R', 'high')
    gh = cv2.getTrackbarPos('G', 'high')
    bh = cv2.getTrackbarPos('B', 'high')
    sh = cv2.getTrackbarPos(switch2, 'high')

    if sh == 0:
        img2[:] = 0
    else:
        img2[:] = [bh, gh, rh]
    # Capture frame-by-frame
    ret, frame = cap.read()

    if frame is not None:
        frame = cv2.medianBlur(frame, 5)
        cv2.imshow('frame', frame)
        # Our operations on the frame come here
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #cv2.imshow('hsv', hsv)

        # hsv = cv2.bilateralFilter(hsv, 10, 75, 270)
        #mask = cv2.inRange(frame, wl, wh)
        mask = cv2.inRange(hsv, np.array([b, g, r]), np.array([bh, gh, rh]))
        mask = cv2.GaussianBlur(mask , (21, 21), 0)
        cv2.imshow('mask', mask)




        # maskw = cv2.inRange(frame, wl, wh)
        # mask = cv2.bilateralFilter(mask, 10, 75, 100)
        # mask = cv2.addWeighted(mask, 0.5, maskw, 0.5, 0.0)

        gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        print(gray.shape[::-1])

        res = cv2.bitwise_and(gray, gray, mask=mask)

        if firstFrame is None:
            firstFrame = gray
            continue

        #frameDelta = cv2.absdiff(firstFrame, gray)
        #frameDelta = cv2.adaptiveThreshold(frameDelta, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                             cv2.THRESH_BINARY_INV, 25, 2)
        cv2.imshow('delta', res)

        # Display the resulting frame
        #cv2.imshow('gray', gray)

        cv2.imshow('hsv', hsv)
        res = cv2.GaussianBlur(res, (17, 17), 0)
        inputImage = res
        rows = inputImage.shape[0]
        circles = cv2.HoughCircles(inputImage, cv2.HOUGH_GRADIENT, 1, rows * 2, param1=60, param2=50, minRadius=30,
                                   maxRadius=700)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            i = circles[0, 0]
            center = (i[0], i[1])
            cv2.arrowedLine(frame, prev, center, (255, 255, 0), 3)
            cv2.circle(frame, center, 1, (0, 100, 100), 3)
            radius = i[2]
            cv2.circle(frame, center, radius, (0, 255, 255), 3)
            cv2.arrowedLine(frame, (640, 360), center, (0, 0, 255), 3)
            x = int(prev[0]) - int(center[0])
            y = int(prev[1]) - int(center[1])
            x = -x
            y = -y
            print(x)
            print(y)
            next = (int(center[0]) + x, int(center[1]) + y)
            prev = center
            cv2.arrowedLine(frame, center, next, (0, 255, 0), 3)
            cv2.imshow('frame', frame)

    key = cv2.waitKey(50)

cv2.destroyAllWindows()