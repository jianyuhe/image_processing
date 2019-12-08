# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:40:00 2019

@author: Cici Liu
"""


import cv2
import numpy as np
import easygui
import math
# read template letters image
letter = cv2.imread('data.png',0)
# create mask image
adpthresh =cv2.adaptiveThreshold(letter, maxValue= 255,
                                 adaptiveMethod= cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 thresholdType= cv2.THRESH_BINARY,
                                  blockSize= 1001,C = 5)
# create mser
mser = cv2.MSER_create()
# find rect of each letters
regions, rects = mser.detectRegions(adpthresh)

# create image of rect of letters then write into template foler
for i in range(len(rects)):
    x = rects[i][0]
    y = rects[i][1]
    w = rects[i][2]
    h = rects[i][3]
    cv2.rectangle(letter, (x, y), (x+w, y+h), color=(0, 0, 255), thickness=1)
    cv2.imwrite("template\\%d.png" %i, letter[y:y+h, x:x+w])

cv2.imshow('thresh',letter)
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows()