import cv2
import numpy as np
import serial
# import serial.tools.list_ports
import glob


def nothing(x):
    pass


string = ''


cap = cv2.VideoCapture(0)
key = ord('-')
prev = (320, 240)
next = prev
while key != ord('q'):

    ret, frame = cap.read()


    if frame is not None:
        frame = cv2.flip(frame, 1)
        cv2.imshow('frame', frame)
    key = cv2.waitKey(60)

    if key == ord('v'):
        while key != ord('q'):
            cap = cv2.VideoCapture(1)
            ret, img = cap.read()
            if img is not None:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

                objp = np.zeros((9 * 6, 3), np.float32)
                objp[:, :2] = np.mgrid[0:6, 0:9].T.reshape(-1, 2)

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                print(gray.shape[::-1])
                # Find the chess board corners
                ret, corners = cv2.findChessboardCorners(gray, (6, 9), None)
                # If found, add object points, image points (after refining them)
                if corners is not None:
                    if ret:
                        objpoints.append(objp)
                        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                        imgpoints.append(corners2)

                    # Draw and display the corners
                    cv2.drawChessboardCorners(img, (6, 9), corners2, ret)
                cv2.imshow('img', img)
                key = cv2.waitKey(2000)
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            np.savez("calib", ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
            cv2.destroyAllWindows()

    if key == ord('c'):

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Arrays to store object points and image points from all the images.
        imgpoints = []  # 2d points in image plane.
        objpoints = []  # 3d point in real world space

        images = glob.glob('chessboards/*.jpg')
        for fname in images:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

            objp = np.zeros((9 * 6, 3), np.float32)
            objp[:, :2] = np.mgrid[0:6, 0:9].T.reshape(-1, 2)

            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print(gray.shape[::-1])
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (6, 9), None)
            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                imgpoints.append(corners2)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (6, 9), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        np.savez("calib", ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
        cv2.destroyAllWindows()

    if key == ord('a'):

        ser = serial.Serial('COM10')

        file = np.load('calib.npz')
        mtx = file['mtx']
        dist = file['dist']

        img = np.zeros((300, 512, 3), np.uint8)
        img2 = np.zeros((300, 512, 3), np.uint8)
        cv2.namedWindow('low')
        cv2.namedWindow('high')

        cv2.createTrackbar('r', 'low', 0, 255, nothing)
        cv2.createTrackbar('g', 'low', 0, 255, nothing)
        cv2.createTrackbar('b', 'low', 0, 255, nothing)
        cv2.createTrackbar('R', 'high', 0, 255, nothing)
        cv2.createTrackbar('G', 'high', 0, 255, nothing)
        cv2.createTrackbar('B', 'high', 0, 255, nothing)

        # create switch for ON/OFF functionality
        switch = '0 : OFF \n1 : ON'
        switch2 = '0 : OFF \n1 : ON'

        cv2.createTrackbar(switch, 'low', 0, 1, nothing)
        cv2.createTrackbar(switch2, 'high', 0, 1, nothing)
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
                frame = cv2.flip(frame, 1)
                frame = cv2.undistort(frame, mtx, dist)
                frame = cv2.medianBlur(frame, 5)
                cv2.imshow('frame', frame)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                # cv2.imshow('hsv', hsv)
                # hsv = cv2.bilateralFilter(hsv, 10, 75, 270)
                # mask = cv2.inRange(frame, wl, wh)
                mask = cv2.inRange(hsv, np.array([b, g, r]), np.array([bh, gh, rh]))
                mask = cv2.GaussianBlur(mask, (21, 21), 0)
                # cv2.imshow('mask', mask)

                gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)

                gray = cv2.medianBlur(gray, 5)
                res = cv2.bitwise_and(gray, gray, mask=mask)
                # cv2.imshow('res', res)
                # Display the resulting frame
                # cv2.imshow('hsv', hsv)
                qres = cv2.GaussianBlur(res, (17, 17), 0)
                inputImage = res
                rows = inputImage.shape[0]
                circles = cv2.HoughCircles(inputImage, cv2.HOUGH_GRADIENT, 1, rows * 2, param1=60, param2=50,
                                           minRadius=30,
                                           maxRadius=700)
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    i = circles[0, 0]
                    center = (i[0], i[1])
                    cv2.arrowedLine(frame, prev, center, (255, 255, 0), 3)
                    cv2.circle(frame, center, 1, (0, 100, 100), 3)
                    radius = i[2]
                    movx = center[0] / 8.77
                    movy = center[1] / 9
                    if movx < 10: string += '0'
                    string += ascii(int(movx))
                    if movy < 10: string += '0'
                    string += ascii(int(movy))
                    print(string)

                    data_to_send = string.encode()
                    print(data_to_send)
                    print(ser.write(data_to_send))
                    # print(ser.write(ascii(int(movx)).encode()))
                    string = ''
                    cv2.circle(frame, center, radius, (0, 255, 255), 3)
                    cv2.arrowedLine(frame, (320, 240), center, (0, 0, 255), 3)
                    x = int(prev[0]) - int(center[0])
                    y = int(prev[1]) - int(center[1])
                    x = -x
                    y = -y
                    next = (int(center[0]) + x, int(center[1]) + y)
                    prev = center
                    cv2.arrowedLine(frame, center, next, (0, 255, 0), 3)
                    cv2.imshow('frame', frame)

            key = cv2.waitKey(30)
    if key == ord('m'):

        center = (320, 240)
        ser = serial.Serial('COM10')

        file = np.load('calib.npz')
        mtx = file['mtx']
        dist = file['dist']

        while key != ord('q'):
            x = center[0]
            y = center[1]
            if key == ord('o'):
                center = (x, y - 8)
            if key == ord('l'):
                center = (x, y + 8)
            if key == ord('k'):
                center = (x - 8, y)
            if key == ord(';'):
                center = (x + 8, y)
            ret, frame = cap.read()
            frame = cv2.undistort(frame, mtx, dist)
            if frame is not None:
                frame = cv2.flip(frame, 1)

                if key == ord('0'):
                    movx = center[0] / 8.77
                    movy = center[1] / 9
                    if movx < 10 : string+='0'
                    string += ascii(int(movx))
                    if movy < 10: string += '0'
                    string += ascii(int(movy))
                    # print(string)
                    data_to_send = string.encode()
                    # print(data_to_send)
                    print(ser.write(data_to_send))
                   # print(ser.write(ascii(int(movx)).encode()))
                    print('BANG!')
                string = ''
                cv2.circle(frame, center, 10, (0, 0, 255), 2)
                cv2.imshow('frame', frame)

            key = cv2.waitKey(50)
    # if key == ord('s'):
    #     lewa = prawa = gora = dol = None
    #     ser = serial.Serial('COM10')
    #     center = (0, 0)
    #     file = np.load('calib.npz')
    #     mtx = file['mtx']
    #     dist = file['dist']
    #     x = 75;
    #     y = 75;
    #     i = 0
    #     while key != ord('q') or i != 8:
    #
    #         # Capture frame-by-frame
    #         if key == ord('o'):
    #             center = (x, y - 10)
    #         if key == ord('l'):
    #             center = (x, y + 10)
    #         if key == ord('k'):
    #             center = (x - 10, y)
    #         if key == ord(';'):
    #             center = (x + 10, y)
    #         ret, frame = cap.read()
    #         frame = cv2.undistort(frame, mtx, dist)
    #         if frame is not None:
    #             frame = cv2.flip(frame, 1)
    #
    #             if key == ord('`'):
    #                 movx = center[0]
    #                 movy = center[1]
    #                 string += ascii(int(movx))
    #                 string += ascii(int(movy))
    #                 # print(string)
    #                 data_to_send = string.encode()
    #                 print(data_to_send)
    #                 print(ser.write(data_to_send))
    #                 # print(ser.write(ascii(int(movx)).encode()))
    #             string = ''
    #
    #             cv2.imshow('frame', frame)
    #         if (i == 1):
    #             lewa = x
    #             i += 1
    #         if (i == 3):
    #             gora = y
    #             i += 1
    #         if(i == 5):
    #             prawa = x
    #             i += 1
    #         if(i == 7):
    #             dol = y
    #             i += 1
    #
    #         key = cv2.waitKey(50)
    #     vertical = dol - gora
    #     horizontal = prawa - lewa
    #     file = open('fov.txt', 'w')
    #
    #     file.write(str(horizontal)+'\n'+str(vertical))


# ser.write('0000'.encode())
cv2.destroyAllWindows()