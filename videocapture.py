#!/usr/bin/env python
from __future__ import print_function
import cv2
import cv2.cv as cv
import os
import datetime

# Read cascade files
cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

def movement(ref, img):
    diff = cv2.absdiff(ref, img)
    retval_diff, dst = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
    return dst


def to_gray(img):
    gray = cv2.cvtColor(img, cv.CV_BGR2GRAY)
    return gray #cv2.equalizeHist(gray)


def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.1,
                                     minNeighbors=3, minSize=(10, 10),
                                     flags = cv.CV_HAAR_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects


def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


log_file = 'faces/capture_log.txt'

if __name__ == '__main__':
    rpath = 'npipe'
    capture = cv2.VideoCapture(rpath)

    ret, img = capture.read()
    ref_gray = to_gray(img)

    cnt = 1
    while True:
        ret, img = capture.read()

        gray = to_gray(img)
        mov = movement(ref_gray, gray).sum()
        if mov > 100:
            print('movement:', end=' ')
            rects = detect(gray, cascade)

            if len(rects) > 0:
                print('rosto')
                with open(log_file, 'a') as f:
                    print(cnt, '-', datetime.datetime.now(), file=f)

                draw_rects(img, rects, (0, 255, 0))
                cv.SaveImage('faces/%05d.png' % cnt, cv.fromarray(img))
                cnt += 1
            else:
                print('no')
        #else:
        #    print('No movement detected: %s' % mov)

        ref_gray = gray
        cv2.imshow('Video', img)
        if 0xFF & cv2.waitKey(1) == 27:
            break
cv2.destroyAllWindows()

