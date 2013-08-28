#!/usr/bin/env python
import cv2
import cv2.cv as cv
import os

# Read cascade files
cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10), flags = cv.CV_HAAR_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

if __name__ == '__main__':
    rpath = 'npipe'
    capture = cv2.VideoCapture(rpath)
    capture.set(cv.CV_CAP_PROP_FRAME_WIDTH, 320)
    capture.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 240)
    while True:
        ret, img = capture.read()

        gray = cv2.cvtColor(img, cv.CV_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        rects = detect(gray, cascade)

        if(len(rects) == 4):
            print 'face detected'
            ## Extract face coordinates         
            x1 = rects[0][1]
            y1 = rects[0][0]
            x2 = rects[0][3]
            y2 = rects[0][2]
            draw_rects(img, rects, (0, 255, 0))
        elif(len(rects) > 0):
            print 'rosto!'
            draw_rects(img, rects, (0, 255, 0))
        else:
            print 'no'

        cv2.imshow('some', img)
        if 0xFF & cv2.waitKey(5) == 27:
            break
cv2.destroyAllWindows()

