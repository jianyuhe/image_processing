import cv2
import numpy as np 
from matplotlib import pyplot as plt


img = cv2.imread("Googly.jpg",0)
img2 = cv2.imread("sudoku.jpg",0)
T, B =cv2.threshold(img, thresh = 100, maxval= 255,type = cv2.THRESH_BINARY)
cv2.imshow('googly', B)

s =cv2.adaptiveThreshold(img2, maxValue= 255,adaptiveMethod= cv2.ADAPTIVE_THRESH_GAUSSIAN_C,thresholdType= cv2.THRESH_BINARY,blockSize= 7,C = 15)
cv2.imshow('sudoku', s) 
 
cv2.waitKey(0)
