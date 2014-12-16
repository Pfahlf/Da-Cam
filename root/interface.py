from tkinter import *
import numpy as np
import cv2
import glob
import sys
import random
import argparse
       
def face():
        cv2.destroyAllWindows()
        cam = cv2.VideoCapture(0)
        name = 'Face detection - DaBuggers'
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
        nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
        pair_eye_cascade = cv2.CascadeClassifier('haarcascade_mcs_eyepair_big.xml'  )

        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        while True:
            s, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                
            eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in eyes:
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)

            pair_eye = pair_eye_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in pair_eye:
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(155,50,255),2)


            cv2.imshow(name, img)    
            k = cv2.waitKey(1)
            if k == 27:
                cv2.destroyAllWindows()
                break
KEY_S = 115
KEY_D = 100

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def drawGrid(img, imgpoints):
    imgpoints = np.int32(imgpoints).reshape(-1,2)
    img = cv2.line(img, tuple(imgpoints[0]), tuple(imgpoints[6]), (255,0,255),5)
    img = cv2.line(img, tuple(imgpoints[0]), tuple(imgpoints[42]), (255,0,255),5)
    img = cv2.line(img, tuple(imgpoints[42]), tuple(imgpoints[48]), (255,0,255),5)
    img = cv2.line(img, tuple(imgpoints[48]), tuple(imgpoints[6]), (255,0,255),5)
    img = cv2.line(img, tuple(imgpoints[4]), tuple(imgpoints[46]), (255,0,255),5)
    img = cv2.line(img, tuple(imgpoints[2]), tuple(imgpoints[44]), (255,0,255),5)
    img = cv2.line(img, tuple(imgpoints[34]), tuple(imgpoints[28]), (255,0,255),5)
    img = cv2.line(img, tuple(imgpoints[20]), tuple(imgpoints[14]), (255,0,255),5)
    return img

def drawLightsaber(img, imgpoints):
    imgpoints = np.int32(imgpoints).reshape(-1,2)
    coordinates = imgpoints[24]
    for i in range(1, 650):
        imgpoints[24][1] -= 1
        img = cv2.circle(img, tuple(imgpoints[24]), 14, (0, 0, 255), -1)
    imgpoints[24][1] = imgpoints[24][1] + 800
    for i in range(1, 200):
        imgpoints[24][1] -= 1
        img = cv2.circle(img, tuple(imgpoints[24]), 19, (200, 200, 200), -1)
    imgpoints[24][1] = imgpoints[24][1] + 200
    for i in range(1, 40):
        imgpoints[24][1] -= 5
        img = cv2.circle(img, tuple(imgpoints[24]), 20, (110, 110, 110), 1)
    imgpoints[24][1] = imgpoints[24][1] + 200
    for i in range(1, 50):
        imgpoints[24][1] -= 4
        img = cv2.circle(img, tuple(imgpoints[24]), 20, (20, 20, 20), 1)
    x = imgpoints[24][0] - 17
    y = imgpoints[24][1] - 20
    a = imgpoints[24][0] - 17
    b = imgpoints[24][1] - 650
    img = cv2.line(img, (x, y), (a,b), (0,0,250),1)
    x = imgpoints[24][0] + 17
    y = imgpoints[24][1] - 20
    a = imgpoints[24][0] + 17
    b = imgpoints[24][1] - 650
    img = cv2.line(img, (x, y), (a,b), (0,0,250),1)
    return img

def drawJail(img):
    for x in range(1500, -700, -100): 
        img = cv2.line(img, (x,900), (x,-200), (40,40,40),15)
    for y in range(900, -200, -100): 
        img = cv2.line(img, (1500,y), (-700,y), (40,40,40),15)

def drawLogo(img, imgpoints):
    imgpoints = np.int32(imgpoints).reshape(-1,2)
    img = cv2.rectangle(img, tuple(imgpoints[0]-100), tuple(imgpoints[24]), (0,0,0),-1)
    img = cv2.rectangle(img, (imgpoints[3][0], imgpoints[3][1]-100), (imgpoints[27][0]+100, imgpoints[27][1]), (255,255,255), -1)
    img = cv2.rectangle(img, (imgpoints[21][0]-100, imgpoints[21][1]), tuple(imgpoints[45]+100), (255,255,255), -1)
    img = cv2.rectangle(img, tuple(imgpoints[24]), tuple(imgpoints[48]+100), (0,0,0), -1)
    img = cv2.circle(img, tuple(imgpoints[24]), 200, (50,50,50), 14)
    img = cv2.circle(img, tuple(imgpoints[24]), 192, (230,230,230), 2)
    img = cv2.circle(img, tuple(imgpoints[24]), 176, (30,30,30), 30)
    img = cv2.circle(img, tuple(imgpoints[24]), 160, (230,230,80), -1)
    font = cv2.FONT_HERSHEY_DUPLEX
    img = cv2.putText(img, "DB", (imgpoints[24][0]-140, imgpoints[24][1]+65), font, 7, (220,200,200), 10)
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
    font = cv2.FONT_HERSHEY_DUPLEX
    if ret == False:
        cv2.putText(img, "No chessboard detected", (20, 50), font, 1, (37,201,11), 2)
    if ret == True:
        cv2.putText(img, "Available keys:", (20, 50), font, 1, (37,201,11), 2)
        cv2.putText(img, "a: show corners of chessboard", (20, 90), font, 1, (37,201,11), 1)
        cv2.putText(img, "d: show grid", (20, 130), font, 1, (37,201,11), 1)
        cv2.putText(img, "f: show lightsaber", (20, 170), font, 1, (37,201,11), 1)
        cv2.putText(img, "g: draw jail", (20, 210), font, 1, (37,201,11), 1)
        cv2.putText(img, "h: show logo", (20, 250), font, 1, (37,201,11), 1)
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        key = cv2.waitKey(1)
        if (key == 97):
            cv2.drawChessboardCorners(img, (7,7), corners, ret)
        elif (key == 100):
            drawGrid(img, imgpoints)
        elif (key == 102):
            drawLightsaber(img, imgpoints)
        elif (key == 103):
            drawJail(img)
        elif (key == 104):
            drawLogo(img, imgpoints)
        # Calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        # Undistortion
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
    return img

def main1():
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
        image = getCameraCalibration(image, objp)
        blur = cv2.GaussianBlur(image,(0,0),1)
        cv2.imshow("2D IMAGES - Da Buggers",blur)

        k = cv2.waitKey(1)
        if k == 27:
                cv2.destroyAllWindows()
                break
        elif player == None:
            key = cv2.waitKey(1)
        elif key == KEY_S:
            saveCalibration()
        elif key == KEY_D:
            displayInfo+=1

def twoD():
        cv2.destroyAllWindows()
        main1()
KEY_S = 115
KEY_D = 100

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def drawPyramid(img, imgpoints):
    imgpoints = np.int32(imgpoints).reshape(-1,2)
##    img = cv2.line(img, tuple(imgpoints[0]), tuple(imgpoints[6]), (255, 0, 0),5)
##    img = cv2.line(img, tuple(imgpoints[0]), tuple(imgpoints[42]), (255, 0, 0),6)
##    img = cv2.line(img, tuple(imgpoints[42]), tuple(imgpoints[48]), (255, 0, 0),6)
##    img = cv2.line(img, tuple(imgpoints[48]), tuple(imgpoints[6]), (255, 0, 0),6)
##    img = cv2.line(img, tuple(imgpoints[0]), tuple(imgpoints[-34]-300,), (255, 0, 0),6)
##    img = cv2.line(img, tuple(imgpoints[6]), tuple(imgpoints[-34]-300), (255, 0, 0),6)
##    img = cv2.line(img, tuple(imgpoints[42]), tuple(imgpoints[-34]-300), (255, 0, 0),6)
##    img = cv2.line(img, tuple(imgpoints[48]), tuple(imgpoints[-34]-300), (255, 0, 0),6)
    lst = []
    coordinates = list(imgpoints[24])
    x = coordinates[0]
    y = coordinates[1]-300
    lst.append(x)
    lst.append(y)
    triangle1 = np.array([ list(imgpoints[0]), list(imgpoints[42]), lst ], np.int32)
    triangle2 = np.array([ list(imgpoints[0]), list(imgpoints[6]), lst ], np.int32)
    triangle3 = np.array([ list(imgpoints[42]), list(imgpoints[48]), lst ], np.int32)
    triangle4 = np.array([ list(imgpoints[48]), list(imgpoints[6]), lst ], np.int32)
##    rectangle = np.array([ list(imgpoints[0]), list(imgpoints[6]), list(imgpoints[42]), list(imgpoints[48]) ], np.int32)
    img = cv2.fillConvexPoly(img, triangle1,(0, 0, 255))
    img = cv2.fillConvexPoly(img, triangle2,(0, 0, 0))
    img = cv2.fillConvexPoly(img, triangle3,(255, 0, 0))
    img = cv2.fillConvexPoly(img, triangle4,(0, 50, 0))
##    img = cv2.fillConvexPoly(img, rectangle, (255, 255, 255))
    return img

def drawShapes(img, imgpoints):
    imgpoints = np.int32(imgpoints).reshape(-1,2)
    for x in range(0, 7, 2):
        img = cv2.line(img, (imgpoints[x][0], imgpoints[x][1]), (imgpoints[x][0], imgpoints[x][1]-300), (40, 40, 40), 15)
    for x in range(1, 7, 2):
        img = cv2.line(img, (imgpoints[x][0], imgpoints[x][1]), (imgpoints[x][0], imgpoints[x][1]-300), (20, 20, 20), 15)
    for x in range(6, 48, 14):
        img = cv2.line(img, (imgpoints[x][0], imgpoints[x][1]), (imgpoints[x][0], imgpoints[x][1]-300), (40, 40, 40), 15)
        img = cv2.line(img, (imgpoints[x][0]-5, imgpoints[x][1]), (imgpoints[x][0]-5, imgpoints[x][1]-300), (40, 40, 40), 15)
    for x in range(20, 48, 14):
        img = cv2.line(img, (imgpoints[x][0], imgpoints[x][1]), (imgpoints[x][0], imgpoints[x][1]-300), (20, 20, 20), 15)
        img = cv2.line(img, (imgpoints[x][0]-5, imgpoints[x][1]), (imgpoints[x][0]-5, imgpoints[x][1]-300), (20, 20, 20), 15)
    for x in range(42, 49, 2):
        img = cv2.line(img, (imgpoints[x][0], imgpoints[x][1]), (imgpoints[x][0], imgpoints[x][1]-300), (40, 40, 40),15)
    for x in range(43, 49, 2):
        img = cv2.line(img, (imgpoints[x][0], imgpoints[x][1]), (imgpoints[x][0], imgpoints[x][1]-300), (20, 20, 20),15)
    for x in range(42, 6, -14):
        img = cv2.line(img, (imgpoints[x][0], imgpoints[x][1]), (imgpoints[x][0], imgpoints[x][1]-300), (40, 40, 40), 15)
        img = cv2.line(img, (imgpoints[x][0]-5, imgpoints[x][1]), (imgpoints[x][0]-5, imgpoints[x][1]-300), (40, 40, 40), 15)
    for x in range(28, 6, -14):
        img = cv2.line(img, (imgpoints[x][0], imgpoints[x][1]), (imgpoints[x][0], imgpoints[x][1]-300), (20, 20, 20), 15)
        img = cv2.line(img, (imgpoints[x][0]-5, imgpoints[x][1]), (imgpoints[x][0]-5, imgpoints[x][1]-300), (20, 20, 20), 15)
    lst = []
    coordinates = list(imgpoints[24])
    x = coordinates[0]
    y = coordinates[1]-400
    lst.append(x)
    lst.append(y)
    triangle1 = np.array([ [imgpoints[0][0], imgpoints[0][1]-300], [imgpoints[42][0], imgpoints[42][1]-300], lst ], np.int32)
    triangle2 = np.array([ [imgpoints[0][0], imgpoints[0][1]-300], [imgpoints[6][0], imgpoints[6][1]-300], lst ], np.int32)
    triangle3 = np.array([ [imgpoints[42][0], imgpoints[42][1]-300], [imgpoints[48][0], imgpoints[48][1]-300], lst ], np.int32)
    triangle4 = np.array([ [imgpoints[48][0], imgpoints[48][1]-300], [imgpoints[6][0], imgpoints[6][1]-300], lst ], np.int32)
    img = cv2.fillConvexPoly(img, triangle1,(0, 0, 0))
    img = cv2.fillConvexPoly(img, triangle2,(40, 40, 40))
    img = cv2.fillConvexPoly(img, triangle3,(40, 40, 40))
    img = cv2.fillConvexPoly(img, triangle4,(0, 0, 0))
    return img

def drawCity(img, imgpoints):
        imgpoints = np.int32(imgpoints).reshape(-1,2)
        # Palace1
        lst1 = []
        coord = list(imgpoints[2])
        x = coord[0]
        y = coord[1]-90
        lst1.append(x)
        lst1.append(y)
        lst2 = []
        coord = list(imgpoints[16])
        x = coord[0]
        y = coord[1]-90
        lst2.append(x)
        lst2.append(y)
        lst3 = []
        coord = list(imgpoints[0])
        x = coord[0]
        y = coord[1]-90
        lst3.append(x)
        lst3.append(y)
        lst4 = []
        coord = list(imgpoints[14])
        x = coord[0]
        y = coord[1]-90
        lst4.append(x)
        lst4.append(y)

        rect1 = np.array([ list(imgpoints[0]), list(imgpoints[2]), lst1,lst3], np.int32)
        rect2 = np.array([ list(imgpoints[0]), list(imgpoints[14]), lst4,lst3], np.int32)
        rect3 = np.array([ list(imgpoints[14]), list(imgpoints[16]),lst2, lst4], np.int32)
        rect4 = np.array([ list(imgpoints[2]), list(imgpoints[16]), lst2,lst1], np.int32)
        rect5 = np.array([ lst3, lst4,lst2,lst1], np.int32)

        img = cv2.fillConvexPoly(img, rect1,(107,147,247))
        img = cv2.fillConvexPoly(img, rect2,(107,147,247))
        img = cv2.fillConvexPoly(img, rect3,(107,147,247))
        img = cv2.fillConvexPoly(img, rect4,(107,147,247))
        img = cv2.fillConvexPoly(img, rect5,(19,121,240))

        # Palace2
        lst1 = []
        coord = list(imgpoints[4])
        x = coord[0]
        y = coord[1]-90
        lst1.append(x)
        lst1.append(y)
        lst2 = []
        coord = list(imgpoints[18])
        x = coord[0]
        y = coord[1]-90
        lst2.append(x)
        lst2.append(y)
        lst3 = []
        coord = list(imgpoints[6])
        x = coord[0]
        y = coord[1]-90
        lst3.append(x)
        lst3.append(y)
        lst4 = []
        coord = list(imgpoints[20])
        x = coord[0]
        y = coord[1]-90
        lst4.append(x)
        lst4.append(y)

        rect1 = np.array([ list(imgpoints[4]), list(imgpoints[6]), lst3,lst1], np.int32)
        rect2 = np.array([ list(imgpoints[4]), list(imgpoints[18]), lst2,lst1], np.int32)
        rect3 = np.array([ list(imgpoints[18]), list(imgpoints[20]),lst4, lst2], np.int32)
        rect4 = np.array([ list(imgpoints[6]), list(imgpoints[20]), lst4,lst3], np.int32)
        rect5 = np.array([ lst1, lst2,lst4,lst3], np.int32)

        img = cv2.fillConvexPoly(img, rect1,(95,223,255))
        img = cv2.fillConvexPoly(img, rect2,(95,223,255))
        img = cv2.fillConvexPoly(img, rect3,(95,223,255))
        img = cv2.fillConvexPoly(img, rect4,(95,223,255))
        img = cv2.fillConvexPoly(img, rect5,(0,255,0))

        # Palace3
        lst1 = []
        coord = list(imgpoints[16])
        x = coord[0]
        y = coord[1]-150
        lst1.append(x)
        lst1.append(y)
        lst2 = []
        coord = list(imgpoints[30])
        x = coord[0]
        y = coord[1]-150
        lst2.append(x)
        lst2.append(y)
        lst3 = []
        coord = list(imgpoints[18])
        x = coord[0]
        y = coord[1]-150
        lst3.append(x)
        lst3.append(y)
        lst4 = []
        coord = list(imgpoints[32])
        x = coord[0]
        y = coord[1]-150
        lst4.append(x)
        lst4.append(y)

        rect1 = np.array([ list(imgpoints[16]), list(imgpoints[18]), lst3,lst1], np.int32)
        rect2 = np.array([ list(imgpoints[16]), list(imgpoints[30]), lst2,lst1], np.int32)
        rect3 = np.array([ list(imgpoints[18]), list(imgpoints[32]),lst4, lst3], np.int32)
        rect4 = np.array([ list(imgpoints[30]), list(imgpoints[32]), lst4,lst2], np.int32)
        rect5 = np.array([ lst1, lst2,lst4,lst3], np.int32)

        img = cv2.fillConvexPoly(img, rect1,(185, 217, 247))
        img = cv2.fillConvexPoly(img, rect2,(185, 217, 247))
        img = cv2.fillConvexPoly(img, rect3,(185, 217, 247))
        img = cv2.fillConvexPoly(img, rect4,(185, 217, 247))
        img = cv2.fillConvexPoly(img, rect5,(69,88,120))

        # Palace4
        lst1 = []
        coord = list(imgpoints[28])
        x = coord[0]
        y = coord[1]-120
        lst1.append(x)
        lst1.append(y)
        lst2 = []
        coord = list(imgpoints[42])
        x = coord[0]
        y = coord[1]-120
        lst2.append(x)
        lst2.append(y)
        lst3 = []
        coord = list(imgpoints[30])
        x = coord[0]
        y = coord[1]-120
        lst3.append(x)
        lst3.append(y)
        lst4 = []
        coord = list(imgpoints[44])
        x = coord[0]
        y = coord[1]-120
        lst4.append(x)
        lst4.append(y)

        rect1 = np.array([ list(imgpoints[28]), list(imgpoints[30]), lst3,lst1], np.int32)
        rect2 = np.array([ list(imgpoints[28]), list(imgpoints[42]), lst2,lst1], np.int32)
        rect3 = np.array([ list(imgpoints[42]), list(imgpoints[44]),lst4, lst2], np.int32)
        rect4 = np.array([ list(imgpoints[30]), list(imgpoints[44]), lst4,lst3], np.int32)
        rect5 = np.array([ lst1, lst2,lst4,lst3], np.int32)

        img = cv2.fillConvexPoly(img, rect1,(139,158,189))
        img = cv2.fillConvexPoly(img, rect2,(139,158,189))
        img = cv2.fillConvexPoly(img, rect3,(139,158,189))
        img = cv2.fillConvexPoly(img, rect4,(139,158,189))
        img = cv2.fillConvexPoly(img, rect5,(15,45,120))

        # Palace5
        lst1 = []
        coord = list(imgpoints[32])
        x = coord[0]
        y = coord[1]-120
        lst1.append(x)
        lst1.append(y)
        lst2 = []
        coord = list(imgpoints[46])
        x = coord[0]
        y = coord[1]-120
        lst2.append(x)
        lst2.append(y)
        lst3 = []
        coord = list(imgpoints[34])
        x = coord[0]
        y = coord[1]-120
        lst3.append(x)
        lst3.append(y)
        lst4 = []
        coord = list(imgpoints[48])
        x = coord[0]
        y = coord[1]-120
        lst4.append(x)
        lst4.append(y)

        rect1 = np.array([ list(imgpoints[32]), list(imgpoints[34]), lst3,lst1], np.int32)
        rect2 = np.array([ list(imgpoints[32]), list(imgpoints[46]), lst2,lst1], np.int32)
        rect3 = np.array([ list(imgpoints[46]), list(imgpoints[48]),lst4, lst2], np.int32)
        rect4 = np.array([ list(imgpoints[34]), list(imgpoints[48]), lst4,lst3], np.int32)
        rect5 = np.array([ lst1, lst2,lst4,lst3], np.int32)

        img = cv2.fillConvexPoly(img, rect1,(181,186,163))
        img = cv2.fillConvexPoly(img, rect2,(181,186,163))
        img = cv2.fillConvexPoly(img, rect3,(181,186,163))
        img = cv2.fillConvexPoly(img, rect4,(181,186,163))
        img = cv2.fillConvexPoly(img, rect5,(247,143,73))
        return img

def drawPython(img,imgpoints):
    imgpoints = np.int32(imgpoints).reshape(-1,2)
    lst1 = []
    coord = list(imgpoints[36])
    x = coord[0]
    y = coord[1]-50
    lst1.append(x)
    lst1.append(y)
    lst2 = []
    coord = list(imgpoints[17])
    x = coord[0]
    y = coord[1]-50
    lst2.append(x)
    lst2.append(y)
    lst3 = []
    coord = list(imgpoints[28])
    x = coord[0]
    y = coord[1]-50
    lst3.append(x)
    lst3.append(y)
    lst4 = []
    coord = list(imgpoints[9])
    x = coord[0]
    y = coord[1]-50
    lst4.append(x)
    lst4.append(y)    

    rect1 = np.array([ list(imgpoints[28]), list(imgpoints[36]), lst1,lst3], np.int32)
    rect2 = np.array([ list(imgpoints[28]), list(imgpoints[9]), lst4,lst3], np.int32)
    rect3 = np.array([ list(imgpoints[9]), list(imgpoints[17]),lst2, lst4], np.int32)
    rect4 = np.array([ list(imgpoints[36]), list(imgpoints[17]), lst2,lst1], np.int32)
    rect5 = np.array([ lst3, lst4,lst2,lst1], np.int32)

    img = cv2.fillConvexPoly(img, rect1,(114,204,117))
    img = cv2.fillConvexPoly(img, rect2,(114,204,117))
    img = cv2.fillConvexPoly(img, rect3,(114,204,117))
    img = cv2.fillConvexPoly(img, rect4,(114,204,117))
    img = cv2.fillConvexPoly(img, rect5,(28,84,29))

    lst1 = []
    coord = list(imgpoints[17])
    x = coord[0]
    y = coord[1]-50
    lst1.append(x)
    lst1.append(y)
    lst2 = []
    coord = list(imgpoints[25])
    x = coord[0]
    y = coord[1]-50
    lst2.append(x)
    lst2.append(y)
    lst3 = []
    coord = list(imgpoints[23])
    x = coord[0]
    y = coord[1]-42
    lst3.append(x)
    lst3.append(y)
    lst4 = []
    coord = list(imgpoints[31])
    x = coord[0]
    y = coord[1]-42
    lst4.append(x)
    lst4.append(y)    

    rect1 = np.array([ list(imgpoints[23]), list(imgpoints[17]), lst1,lst3], np.int32)
    rect2 = np.array([ list(imgpoints[23]), list(imgpoints[31]), lst4,lst3], np.int32)
    rect3 = np.array([ list(imgpoints[17]), list(imgpoints[25]),lst2, lst1], np.int32)
    rect4 = np.array([ list(imgpoints[31]), list(imgpoints[25]), lst2,lst4], np.int32)
    rect5 = np.array([ lst3, lst1,lst2,lst4], np.int32)

    img = cv2.fillConvexPoly(img, rect1,(114,204,117))
    img = cv2.fillConvexPoly(img, rect2,(114,204,117))
    img = cv2.fillConvexPoly(img, rect3,(114,204,117))
    img = cv2.fillConvexPoly(img, rect4,(114,204,117))
    img = cv2.fillConvexPoly(img, rect5,(28,84,29))

    lst1 = []
    coord = list(imgpoints[33])
    x = coord[0]
    y = coord[1]-100
    lst1.append(x)
    lst1.append(y)
    lst2 = []
    coord = list(imgpoints[31])
    x = coord[0]
    y = coord[1]-100
    lst2.append(x)
    lst2.append(y)
    lst3 = []
    coord = list(imgpoints[39])
    x = coord[0]
    y = coord[1]-100
    lst3.append(x)
    lst3.append(y)
    lst4 = []
    coord = list(imgpoints[25])
    x = coord[0]
    y = coord[1]-100
    lst4.append(x)
    lst4.append(y)    
  
    rect1 = np.array([ list(imgpoints[39]), list(imgpoints[33]), lst1,lst3], np.int32)
    rect2 = np.array([ list(imgpoints[31]),list(imgpoints[25]), lst4,lst2], np.int32)
    rect3 = np.array([ list(imgpoints[39]), list(imgpoints[31]),lst2, lst1], np.int32)
    rect4 = np.array([ list(imgpoints[25]), list(imgpoints[33]), lst1,lst4], np.int32)
    rect5 = np.array([ lst3, lst1,lst4,lst2], np.int32)

    img = cv2.fillConvexPoly(img, rect1,(114,204,117))
    img = cv2.fillConvexPoly(img, rect2,(114,204,117))
    img = cv2.fillConvexPoly(img, rect3,(114,204,117))
    img = cv2.fillConvexPoly(img, rect4,(114,204,117))
    img = cv2.fillConvexPoly(img, rect5,(28,84,29))    


    lst1 = []
    coord = list(imgpoints[27])
    x = coord[0]
    y = coord[1]-150
    lst1.append(x)
    lst1.append(y)
    lst2 = []
    coord = list(imgpoints[31])
    x = coord[0]
    y = coord[1]-150
    lst2.append(x)
    lst2.append(y)
    lst3 = []
    coord = list(imgpoints[39])
    x = coord[0]
    y = coord[1]-150
    lst3.append(x)
    lst3.append(y)
    lst4 = []
    coord = list(imgpoints[19])
    x = coord[0]
    y = coord[1]-150
    lst4.append(x)
    lst4.append(y)    
    lst5 = []
    coord = list(imgpoints[31])
    x = coord[0]
    y = coord[1]-100
    lst5.append(x)
    lst5.append(y)
    lst6 = []
    coord = list(imgpoints[39])
    x = coord[0]
    y = coord[1]-100
    lst6.append(x)
    lst6.append(y)
    lst7 = []
    coord = list(imgpoints[19])
    x = coord[0]
    y = coord[1]-100
    lst7.append(x)
    lst7.append(y)
    lst8 = []
    coord = list(imgpoints[27])
    x = coord[0]
    y = coord[1]-100
    lst8.append(x)
    lst8.append(y)   
    lst9 = []
    coord = list(imgpoints[39.9])
    x = coord[0]
    y = coord[1]-150
    lst9.append(x)
    lst9.append(y)
    lst10 = []
    coord = list(imgpoints[39.9])
    x = coord[0]
    y = coord[1]-100
    lst10.append(x)
    lst10.append(y)   
    rect1 = np.array([lst5, lst6, lst3,lst2], np.int32)
    rect2 = np.array([ lst5,lst7, lst4,lst2], np.int32)
    rect3 = np.array([ lst6, lst8,lst1, lst3], np.int32)
    rect4 = np.array([ lst7, lst8, lst1,lst4], np.int32)
    rect5 = np.array([ lst3, lst1,lst4,lst2], np.int32)
    img = cv2.fillConvexPoly(img, rect1,(114,204,117))
    img = cv2.fillConvexPoly(img, rect2,(114,204,117))
    img = cv2.fillConvexPoly(img, rect3,(114,204,117))
    img = cv2.fillConvexPoly(img, rect4,(114,204,117))
    img = cv2.fillConvexPoly(img, rect5,(28,84,29))    
    return img

    
def getCameraCalibration1(img, objp):
    global criteria
    objpoints = []
    imgpoints = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret = False
    # Find the corners of the chessboard
    ret, corners = cv2.findChessboardCorners(gray, (7,7), None)
    h,w = img.shape[:2]
    # If found, add object points, image points
    font = cv2.FONT_HERSHEY_DUPLEX
    if ret == False:
        cv2.putText(img, "No chessboard detected", (20, 50), font, 1, (37,201,11), 2)
    if ret == True:
        cv2.putText(img, "Available keys:", (20, 50), font, 1, (37,201,11), 2)
        cv2.putText(img, "a: show pyramid", (20, 90), font, 1, (37,201,11), 1)
        cv2.putText(img, "d: show shapes", (20, 130), font, 1, (37,201,11), 1)
        cv2.putText(img, "f: show city", (20, 170), font, 1, (37,201,11), 1)
        cv2.putText(img, "g: draw python", (20, 210), font, 1, (37,201,11), 1)
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        key = cv2.waitKey(1)
        if (key == 97):
            drawPyramid(img, imgpoints)
        elif (key == 100):
            drawShapes(img, imgpoints)
        elif (key == 102):
            drawCity(img, imgpoints)
        elif (key == 103):
            drawPython(img, imgpoints)
        # Calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        # Undistortion
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
    return img

def main2():
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
        image = getCameraCalibration1(image, objp)
        blur = cv2.GaussianBlur(image,(0,0),1)
        cv2.imshow("3D SHAPES - Da Buggers",blur)

        if player == None:
            key = cv2.waitKey(1)
        
        if (key == 27):
            capture.release()
            cv2.destroyAllWindows()
            break
        elif key == KEY_S:
            saveCalibration()
        elif key == KEY_D:
            displayInfo+=1
        
def treD():
        cv2.destroyAllWindows()
        main2()


KEY_S = 115
KEY_D = 100

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
def printUSI(img, imgpoints):
    imgpoints = np.int32(imgpoints).reshape(-1,2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.putText(img,'USI',(imgpoints[42][0], imgpoints[42][1]), font, 14,(255,255,255),15)
    return img

def printRandomUSI(img, imgpoints):
    imgpoints = np.int32(imgpoints).reshape(-1,2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    z = random.randrange(20, 60, 5)
    for i in range(z):
        c = random.randrange(0, 255, 1)
        x = random.randrange(-500,500,1)
        y = random.randrange(-500,500,1)
        f = random.randrange(1, 15, 1)
        cv2.putText(img,'USI',(imgpoints[0][0]+x, imgpoints[0][1]+y), font, f,(c,c,c),f)

def printFounders(img, imgpoints):
    imgpoints = np.int32(imgpoints).reshape(-1,2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    z = random.randrange(0,5,1)
    for i in range(z):
        z1 = random.randrange(5, 15, 1)
        z2 = random.randrange(5, 15, 1)
        z3 = random.randrange(5, 15, 1)
        z4 = random.randrange(5, 15, 1)
        x1 = random.randrange(-800,600,5)
        x2 = random.randrange(-800,600,5)
        x3 = random.randrange(-800,600,5)
        x4 = random.randrange(-800,600,5)
        y1 = random.randrange(-800,600,5)
        y2 = random.randrange(-800,600,5)
        y3 = random.randrange(-800,600,5)
        y4 = random.randrange(-800,600,5)
        f = random.randrange(1, 10, 1)
        for i in range(z1):
            cv2.putText(img, "Leonardo",(imgpoints[0][0]+x1, imgpoints[0][1]+y1), font, f,(255,255,255),f)
        for i in range(z2):
            cv2.putText(img, "Ersin",(imgpoints[0][0]+x2, imgpoints[0][1]+y2), font, f,(255,255,255),f)
        for i in range(z3):
            cv2.putText(img, "Gabriele",(imgpoints[0][0]+x3, imgpoints[0][1]+y3), font, f,(255,255,255),f)
        for i in range(z4):
            cv2.putText(img, "Federico",(imgpoints[0][0]+x4, imgpoints[0][1]+y4), font, f,(255,255,255),f)

def printNumbers(img, imgpoints):
    imgpoints = np.int32(imgpoints).reshape(-1,2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(20):
        for j in range(100):
            r = random.randrange(0,255,1)
            b = random.randrange(0,255,1)
            g = random.randrange(0,255,1)
            x = random.randrange(-1200,1200,1)
            y = random.randrange(-800,800,1)
            if j <= 100 and j > 90:
                cv2.putText(img,'9',(imgpoints[24][0]+x, imgpoints[24][1]+y), font, 1,(g,b,r),1)
            elif j <= 90 and j > 80:
                cv2.putText(img,'8',(imgpoints[24][0]+x, imgpoints[24][1]+y), font, 1,(g,b,r),1)
            elif j <= 80 and j > 70:
                cv2.putText(img,'7',(imgpoints[24][0]+x, imgpoints[24][1]+y), font, 1,(g,b,r),1)
            elif j <= 70 and j > 60:
                cv2.putText(img,'6',(imgpoints[24][0]+x, imgpoints[24][1]+y), font, 1,(g,b,r),1)
            elif j <= 60 and j > 50:
                cv2.putText(img,'5',(imgpoints[24][0]+x, imgpoints[24][1]+y), font, 1,(g,b,r),1)
            elif j <= 50 and j > 40:
                cv2.putText(img,'4',(imgpoints[24][0]+x, imgpoints[24][1]+y), font, 1,(g,b,r),1)
            elif j <= 40 and j > 30:
                cv2.putText(img,'3',(imgpoints[24][0]+x, imgpoints[24][1]+y), font, 1,(g,b,r),1)
            elif j <= 30 and j > 20:
                cv2.putText(img,'2',(imgpoints[24][0]+x, imgpoints[24][1]+y), font, 1,(g,b,r),1)
            elif j <= 20 and j > 10:
                cv2.putText(img,'1',(imgpoints[24][0]+x, imgpoints[24][1]+y), font, 1,(g,b,r),1)
            elif j <= 10 and j > 0:
                cv2.putText(img,'0',(imgpoints[24][0]+x, imgpoints[24][1]+y), font, 1,(g,b,r),1)

    
def getCameraCalibration3(img, objp):
    global criteria
    objpoints = []
    imgpoints = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret = False
    # Find the corners of the chessboard
    ret, corners = cv2.findChessboardCorners(gray, (7,7), None)
    h,w = img.shape[:2]
    # If found, add object points, img points
    font = cv2.FONT_HERSHEY_DUPLEX
    if ret == False:
        cv2.putText(img, "No chessboard detected", (20, 50), font, 1, (37,201,11), 2)
    if ret == True:
        cv2.putText(img, "Available keys:", (20, 50), font, 1, (37,201,11), 2)
        cv2.putText(img, "a: print 'USI'", (20, 90), font, 1, (37,201,11), 1)
        cv2.putText(img, "d: randomly print 'USI'", (20, 130), font, 1, (37,201,11), 1)
        cv2.putText(img, "f: print names of the founders", (20, 170), font, 1, (37,201,11), 1)
        cv2.putText(img, "g: print random numbers", (20, 210), font, 1, (37,201,11), 1)
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        key = cv2.waitKey(1)
        if (key == 97):
            printUSI(img, imgpoints)
        elif (key == 100):
            printRandomUSI(img, imgpoints)
        elif (key == 102):
            printFounders(img, imgpoints)
        elif (key == 103):
            printNumbers(img, imgpoints)
        # Calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        # Undistortion
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
    return img

def main3():
    player = None
    capture = cv2.VideoCapture(0)
    objp = np.zeros((7*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:21.3:3.55,0:21.3:3.55].T.reshape(-1,2)
    displayInfo = 0
    while capture.isOpened:
        if not player == None:
            key = player.loop()
            img = player.img.copy()
        else:
            f,img = capture.read()
        img = getCameraCalibration3(img, objp)
        blur = cv2.GaussianBlur(img,(0,0),1)
        cv2.imshow("TEXT PRINTER - Da Buggers",blur)

        k = cv2.waitKey(1)
        if k == 27:
                cv2.destroyAllWindows()
                break

        if player == None:
            key = cv2.waitKey(1)
        
        if (key == 27):
            capture.release()
            cv2.destroyAllWindows()
            break
        elif key == KEY_S:
            saveCalibration()
        elif key == KEY_D:
            displayInfo+=1

def text():
        cv2.destroyAllWindows()
        main3()


def intro():
    cam = cv2.VideoCapture(0)
    name = 'VIP DETECTION - DaBuggers'
    a = 'During our tests, the program had an accuracy of 66%.'
    b = 'Press "return" to continue'

    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)

    while True:
        s, img = cam.read()
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(img,(0,0),10)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.putText(blur,a,(320,250), font, 0.7,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(blur,b,(490,520), font, 0.7,(255,255,255),2,cv2.LINE_AA) 


        cv2.imshow(name, blur)

        k = cv2.waitKey(1)
        
        if k == 13:
            break

        if k == 114:
            search()            
        
        if k == 27:
            cv2.destroyAllWindows()
            return
        

   


def search():
    cam = cv2.VideoCapture(0)
    name = 'V.I.P. DETECTION - DaBuggers'
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
    nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
    pair_eye_cascade = cv2.CascadeClassifier('haarcascade_mcs_eyepair_big.xml')

    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)


    Face_size = []

    Eyes_size = []

    Pair_eyes_size = []

    Sub_list_faces = []

    Sub_list_eyes = []

    Sub_list_pair_eyes = []


    Person_temp = [] 
    Person_details = []
    Person_names = ['STEVE JOBS','Dead',164,99,'Steve Wozniak','64',90,163,'Mehdi Jazayeri','Who knows?',210]
    Person_names_print = []


    database_names = []

    count_face = 0
    count_eyes = 0
    count_pair_eyes = 0


    ######LABEL and INSTR#######


    def label():
        label = '-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+'
        label2= '|'


        instructions = 'Instructions: put the face square inside'
        instructions2 = 'the label, press return and wait.'
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(0,800,18):
            cv2.putText(img,label2,(435,-2+i), font, 1,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(img,label2,(840,-2+i), font, 1,(255,255,255),1,cv2.LINE_AA)
        cv2.rectangle(img,(0,0),(600,100),(0,0,0),-1)
        cv2.putText(img,instructions,(10,25), font, 0.7,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(img,instructions2,(10,50), font, 0.7,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(img,label,(0,160), font, 1,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(img,label,(0,600), font, 1,(255,255,255),1,cv2.LINE_AA)

    def label2():
        label = '-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+'
        label2= '|'


        instructions = "Do not move. Calculating... "
        instructions2 = "___________________________________________"
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(0,800,18):
            cv2.putText(img,label2,(435,-2+i), font, 1,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(img,label2,(840,-2+i), font, 1,(255,255,255),1,cv2.LINE_AA)
        cv2.rectangle(img,(0,0),(600,100),(0,0,0),-1)
        cv2.putText(img,instructions,(10,25), font, 0.7,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(img,instructions2,(10,50), font, 0.7,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(img,label,(0,160), font, 1,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(img,label,(0,600), font, 1,(255,255,255),1,cv2.LINE_AA)


    while True:
        s, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        label()

        cv2.imshow(name, img)

        k = cv2.waitKey(1)
        
        if k == 13:
            break

        if k == 114:
            search()            
        
        if k == 27:
            cv2.destroyAllWindows()
            return
        
    while True:
        s, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        label2()
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        ######SEARCH FACE#######    
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)


        for (x,y,w,h) in faces:
            
            Face_size.append(x)
            Face_size.append(y)
            Face_size.append(w)
            Face_size.append(h)
            Sub_list_faces.append(Face_size)
            Face_size = []
            count_face += 1

            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        ######SEARCH MEDIA OF LENGTH FACE#######

        length_face = []

        if count_face == 20:
            for i in range (len(Sub_list_faces)):
        ##            print (Sub_list_faces[i])
                
                if Sub_list_faces[i][2] < 100:
                    print('error,wrong object caught')
                    pass
                else:
                    length_face.append(int(Sub_list_faces[i][2]))

        ##        print(length_face)

            print('Media lungezza faccia = ',np.mean(length_face))

            x = np.mean(length_face)

            Person_temp.append(x)


        ######SEARCH MEDIA OF Large FACE#######

        large_face = []

        if count_face== 20:
            for i in range (len(Sub_list_faces)):
        ##            print (Sub_list_faces[i])
                
                if Sub_list_faces[i][3] < 100:
                    print('error,wrong object caught')
                    pass
                else:
                    large_face.append(int(Sub_list_faces[i][3]))

        ##        print(large_face)

            print('Media larghezza faccia = ',np.mean(large_face))

            x = np.mean(large_face)

            Person_temp.append(x)


        ######SEARCH BOTH EYES#######


        eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in eyes:
            
            Eyes_size.append(x)
            Eyes_size.append(y)
            Eyes_size.append(w)
            Eyes_size.append(h)
            Sub_list_eyes.append(Eyes_size)
            Eyes_size = []
            count_eyes += 1
            
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)

        ######SEARCH MEDIA Squares#######

        large_eyes = []

        if count_eyes == 20:
            for i in range (len(Sub_list_eyes)):
        ##            print (Sub_list_eyes[i])
                
                large_eyes.append(int(Sub_list_eyes[i][2]))

        ##        print(large_eyes)

            print('Media larghezza occhi = ',np.mean(large_eyes))

    ##        x = np.mean(large_eyes)
    ##
    ##        Person_temp.append(x)

        ############################

        ##    mouth = mouth_cascade.detectMultiScale(gray, 1.3, 5)
        ##    for (x,y,w,h) in mouth:
        ##        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        ##    nose = nose_cascade.detectMultiScale(gray, 1.3, 5)
        ##    for (x,y,w,h) in nose:
        ##        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        ######SEARCH RECT EYES#######

        pair_eye = pair_eye_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in pair_eye:

            Pair_eyes_size.append(x)
            Pair_eyes_size.append(y)
            Pair_eyes_size.append(w)
            Pair_eyes_size.append(h)
            Sub_list_pair_eyes.append(Pair_eyes_size)
            Pair_eyes_size = []
            count_pair_eyes += 1
            
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(155,80,255),2)


        ######SEARCH MEDIA PAIR#######

        large_pair_eyes = []

        if count_pair_eyes == 20:
            for i in range (len(Sub_list_pair_eyes)):
        ##            print (Sub_list_pair_eyes[i])
                
                large_pair_eyes.append(int(Sub_list_pair_eyes[i][2]))

            print('Media larghezza rettangolo occhi = ',np.mean(large_pair_eyes))

    ##        x = np.mean(large_pair_eyes)
    ##
    ##        Person_temp.append(x)

        ####PRINT INFO####

        Person_temp_info = []
        load = []

        count = 0
        for i in Person_temp:
            Person_temp_info.append(int(i))
            load.append('...')
            count +=1
            if count == 2:
                load.append('//Press "return"')
            else:
                continue
            

        cv2.putText(img,str(load),(10,75), font, 0.7,(255,255,255),1,cv2.LINE_AA)


        ###### EXIT #######

        cv2.imshow(name, img)

        k = cv2.waitKey(1)

        if k == 13:
            break

        if k == 114:
            search()            
        
        if k == 27:
            cv2.destroyAllWindows()
            return


    for i in Person_temp:
        Person_details.append(int(i))


    ##f = open("data_details", "w")
    ##f.write("\n".join(map(lambda x: str(x), Person)))
    ##f.close()
    ##
    ##list_info = open('Data', 'r')
    ##for line in list_info:
    ##    database_names.append(line.strip())

    det = min(Person_details, key=lambda x:abs(x-180))

    if det < 300 and det > 199:
        Person_names_print.append(Person_names[8])
        Person_names_print.append(Person_names[9])
        
    elif det < 195 and det > 165:
        Person_names_print.append(Person_names[4])
        Person_names_print.append(Person_names[5])

    elif det < 165:
        Person_names_print.append(Person_names[0])
        Person_names_print.append(Person_names[1])


    ##if (180 - det) < (det - 150) and (det - 180) > (det - 200):
    ##    Person_names_print.append(Person_names[8])
    ##    Person_names_print.append(Person_names[9])
    ##
    ##elif (180 - det) < (det - 150) and (det - 180) < (det - 200):
    ##    Person_names_print.append(Person_names[4])
    ##    Person_names_print.append(Person_names[5])
    ##    
    ##else:
    ##    Person_names_print.append(Person_names[0])
    ##    Person_names_print.append(Person_names[1])


    def label3_inf():
        who = "Subject:"
        who_2 = Person_names_print[0]
        age = "Age:"
        age_2 =  Person_names_print[1]

        info = 'press "r" to restart'
        info2 = 'press "esc" to exit'

        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(img,(0,0),(400,107),(0,0,0),-1)
        cv2.putText(img,who,(10,25), font, 0.7,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(img,str(who_2),(180,25), font, 0.7,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(img,age,(10,50), font, 0.7,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(img,str(age_2),(160,50), font, 0.7,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(img,info,(10,75), font, 0.7,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(img,info2,(10,100), font, 0.7,(255,255,255),1,cv2.LINE_AA)



    while True:
        s, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        label3_inf()

        cv2.imshow(name, img)    
        k = cv2.waitKey(1)
        
        if k == 13:
            break

        if k == 114:
            search()            
        
        if k == 27:
            cv2.destroyAllWindows()
            return
        cv2.imshow(name, img)

def vipde():
        cv2.destroyAllWindows()
        intro()
        search()

root = Tk()
root.geometry("%dx%d+%d+%d" % (174, 342, 0, 0))
root.title('Da Cam')


#title
photo = PhotoImage(file="foto.gif")
title = Label(image = photo).grid(sticky=N)
#button
bottone1 = Button(root, text='FACE DETECTION', command = face, padx = 30).grid(sticky=W)
bottone5 = Button(root, text='VIP DETECTION', command = vipde, padx = 37).grid(sticky=W)
bottone2 = Button(root, text='TEXT PRINT', command = text, padx = 48).grid(sticky=W)
bottone3 = Button(root, text='2D IMAGES', command = twoD, padx = 50).grid(sticky=W)
bottone4 = Button(root, text='3D SHAPES', command = treD, padx = 49).grid(sticky=W)

Label().grid()


Copyright = Label(text = '           © Da Buggers').grid(sticky=W)
root.mainloop()




