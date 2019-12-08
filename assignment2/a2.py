##################################################################################
#Image Processing Assignment2
# For Assignment2, I will using python code to segemnet white ball from grass
# and extrac the white ball and replace it by grass
# author = Minhui Chen 
# DT228
# Student No: D17125347
#################################################################################
import numpy as np
import cv2
import easygui
from matplotlib import pyplot as plt

#read image
f = easygui.fileopenbox()
I = cv2.imread(f) 
img = cv2.GaussianBlur(I,(3,3),0)

#convert the image to grayscale
G = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Y = cv2.cvtColor(I, cv2.COLOR_BGR2YUV)
#H = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
#
#Ix = cv2.Sobel(G,ddepth=cv2.CV_64F,dx=1,dy=0)
#Iy = cv2.Sobel(G,ddepth=cv2.CV_64F,dx=0,dy=1)
#

edge=cv2.Canny(G, 240, 250)


lower_white = np.array([0,0,0])
upper_white = np.array([0,0,0])
mask = cv2.inRange(I, lower_white, upper_white)


#mask = cv2.inRange(G, 220,255)
#
#
#kernel = np.ones((3,3), np.uint8)
#mask = cv2.erode(mask, kernel, iterations=2)
#mask = cv2.dilate(mask, kernel, iterations=4)

#houghCircles
#cirecles = cv2.HoughCircles(edge, cv2.HOUGH_GRADIENT,1,50,param1=20,param2=30)
#
#for i in circles[0, :]:
#    cv2.circle(edge, (i[0], i[1]), i[2], (0, 255, 0), 2)  # 画出外圆
#    cv2.circle(edge, (i[0], i[1]), 2, (0, 0, 255), 3)  # 画出圆心


#Threshold


#T = 230
#T, B = cv2.threshold(G, thresh = T, maxval = 255, type = cv2.THRESH_BINARY)



#cv2.imshow('Gray', G)
#cv2.imshow('Y', Y)
#cv2.imshow('H', H)
cv2.imshow('Image', edge)
cv2.imshow('Image', mask)
cv2.waitKey()


