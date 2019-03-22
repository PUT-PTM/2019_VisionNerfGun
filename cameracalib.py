import numpy as np
import cv2
import glob
#
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all the images.
imgpoints = []  # 2d points in image plane.
objpoints = []  # 3d point in real world space

images = glob.glob('*.jpg')
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
    cv2.waitKey(3000)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
np.savez("calib", ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
cv2.destroyAllWindows()
