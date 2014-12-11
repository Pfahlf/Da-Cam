import numpy as np
import cv2
import glob
import sys

KEY_S = 115
KEY_D = 100

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def drawGrid(img, corners, imgpoints):
    imgpoints = np.int32(imgpoints).reshape(-1,2)
    img = cv2.line(img, tuple(imgpoints[0]), tuple(imgpoints[6]), (0,0,255),5)
    img = cv2.line(img, tuple(imgpoints[0]), tuple(imgpoints[42]), (0,0,255),5)
    img = cv2.line(img, tuple(imgpoints[42]), tuple(imgpoints[48]), (0,0,255),5)
    img = cv2.line(img, tuple(imgpoints[48]), tuple(imgpoints[6]), (0,0,255),5)
    img = cv2.line(img, tuple(imgpoints[4]), tuple(imgpoints[46]), (0,0,255),5)
    img = cv2.line(img, tuple(imgpoints[2]), tuple(imgpoints[44]), (0,0,255),5)
    img = cv2.line(img, tuple(imgpoints[34]), tuple(imgpoints[28]), (0,0,255),5)
    img = cv2.line(img, tuple(imgpoints[20]), tuple(imgpoints[14]), (0,0,255),5)
    return img

def drawPyramid(img, corners, imgpoints):
    imgpoints = np.int32(imgpoints).reshape(-1,2)
    img = cv2.line(img, tuple(imgpoints[0]), tuple(imgpoints[6]), (0,0,255),5)
    img = cv2.line(img, tuple(imgpoints[0]), tuple(imgpoints[42]), (0,0,255),5)
    img = cv2.line(img, tuple(imgpoints[42]), tuple(imgpoints[48]), (0,0,255),5)
    img = cv2.line(img, tuple(imgpoints[48]), tuple(imgpoints[6]), (0,0,255),5)
    img = cv2.line(img, tuple(imgpoints[0]), tuple(imgpoints[48]-200), (0,0,255),5)
    img = cv2.line(img, tuple(imgpoints[6]), tuple(imgpoints[48]-200), (0,0,255),5)
    img = cv2.line(img, tuple(imgpoints[42]), tuple(imgpoints[48]-200), (0,0,255),5)
    img = cv2.line(img, tuple(imgpoints[48]), tuple(imgpoints[48]-200), (0,0,255),5)
    return img

def getCameraCalibration(img, objp):
    global criteria
    objpoints = []
    imgpoints = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret = False
    # Find the corners of the chessboard
    ret, corners = cv2.findChessboardCorners(gray, (7,7), None)
    h,w = img.shape[:2]
    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        #cv2.drawChessboardCorners(img, (7,7), corners, ret)
        #drawGrid(img, corners, imgpoints)
        drawPyramid(img, corners, imgpoints)
        # Calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        # Undistortion
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
    return img

def main():
    player = None
    capture = cv2.VideoCapture(0)
    objp = np.zeros((7*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:21.3:3.55,0:21.3:3.55].T.reshape(-1,2)
    displayInfo = 0
    while capture.isOpened:
        if not player == None:
            key = player.loop()
            image = player.image.copy()
        else:
            f,image = capture.read()
        # Qui determino la sfuocatura !!!!!
        image = getCameraCalibration(image, objp)
        blur = cv2.GaussianBlur(image,(0,0),1)
        cv2.imshow("WEBCAM",blur)

        if player == None:
            key = cv2.waitKey(1)
        
        if(key == 27):
            capture.release()
            cv2.destroyAllWindows()
            break
        elif key == KEY_S:
            saveCalibration()
        elif key == KEY_D:
            displayInfo+=1
main()

        
