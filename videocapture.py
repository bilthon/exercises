#!/usr/bin/env python
from __future__ import print_function
import cv2
import cv2.cv as cv
import numpy as np
import os
import datetime
import glob
import re

# Read cascade files
cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")

def movement(ref, img):
    diff = cv2.absdiff(ref, img)
    retval_diff, dst = cv2.threshold(diff, 50, 1, cv2.THRESH_BINARY)
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

font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, .5, .5, lineType=cv.CV_AA)
base_data_dir = 'faces/'
log_file = 'capture_log.txt'
img_ext = 'jpg'
join = os.path.join

def dirname(base_dir=base_data_dir):
    """ Generate dir name in directory for the day (inside base_dir)
        Dir name format: AAAA_MM_DD
        If the dir do not exists, it will be created.
    """
    now = datetime.datetime.now()
    dir = '%02d_%02d_%02d' % (now.year, now.month, now.day)
    dir = join(base_dir, dir, '')
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir
    

if __name__ == '__main__':
    data_dir = dirname()
    imgs = sorted(glob.glob(data_dir + '*.' + img_ext))
    try:
        cnt = int(re.search(re.escape(data_dir) + r'(\d+)', imgs[-1]).group(1)) + 1
    except Exception:
        cnt = 1

    capture = cv2.VideoCapture('/dev/stdin')
    print('Capture started')

    ret, img = capture.read()
    ref_gray = to_gray(img)

    while True:
        ret, img = capture.read()

        gray = to_gray(img)
        mov = movement(ref_gray, gray).sum()
        if mov > 100:
            print('movement:', end=' ')
            rects = detect(gray, cascade)

            if len(rects) > 0:
                print('rosto')

                data_dir_now = dirname()
                if data_dir_now != data_dir:
                    cnt = 1
                    data_dir = data_dir_now

                timestamp = str(datetime.datetime.now())
                with open(join(data_dir, log_file), 'a') as f:
                    print(cnt, '-', timestamp, file=f)

                draw_rects(img, rects, (0, 255, 0))
                final_img = cv.fromarray(img)
                cv.PutText(final_img, timestamp.split('.')[0], (10, 20), font, (0, 0, 255))
                cv.SaveImage(join(data_dir, '%06d.' % cnt + img_ext), final_img)
                cnt += 1
            else:
                print('no')
        #else:
        #    print('No movement detected: %s' % mov)

        ref_gray = gray
        cv2.imshow('Video', img)
        #cv2.imshow('Video', np.asarray(final_img[:,:]))

        if 0xFF & cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()

