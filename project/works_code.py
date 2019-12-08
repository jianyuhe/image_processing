# -*- coding: utf-8 -*-
"""
Created on  Nov 1 14:00:30 2019

@author: Yuwei Chen
"""


import pytesseract
import cv2
import numpy as np
import easygui
import math
from math import *
import time
import os
import re
from matplotlib import pyplot as plt


#morphologe function to get a binary image without nosie(or less noise)
def morphology(grayImg):
    # 1. apply av horizontal Sobel filter
    sobel = cv2.Sobel(grayImg, cv2.CV_8U, 1, 0, ksize = 3)
    
    #get binary image
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
 
    #Remove spots noise by apply morphological operations
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))
    dilation = cv2.dilate(binary, element, iterations = 1)
    cv2.imwrite("dilation.jpg",dilation)
    return  dilation 

#function Point A goes around point B angle degrees counterclockwise in image
def pointRotation(image,angle,pointA,pointB):   
  h,w = image.shape[:2]
  ax = pointA[0]
  ay = h - pointA[1]
  bx = pointB[0]
  by = h - pointB[1]
  
  nRotatex = (ax-bx)*math.cos(math.pi / 180.0 * angle) - (ay-by)*math.sin(math.pi / 180.0 * angle) + bx
  nRotatey = (ax-bx)*math.sin(math.pi / 180.0 * angle) + (ay-by)*math.cos(math.pi / 180.0 * angle) + by
  
  rotatedPoint =(nRotatex, h - nRotatey)
  return rotatedPoint

#findTextRegion function to find the text regions in image, so we can crop these region in order to extrac text better
def findTextRegion(grayImg):
    region = []
    angle = []
    width=[]
    height = []
    #find contours
    contours, hierarchy = cv2.findContours(grayImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 
    #for loop traverse contours and remove all small area of the contourns(small area of the contourns may be spots)
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt) 
        
        if(area < 1000):
            continue
 
        #get the smallest rectangle ( may have a rotation angle)
        rect = cv2.minAreaRect(cnt)
        
        ((cx, cy), (w, h), theta) = rect #get the tilt Angle of the rectangle theta
        
        # box is the coordinates of the four points of the rectangle
        box = cv2.boxPoints(rect)
        box = np.int0(box)
 
        # The length and width of the rectangle
        heightR = abs(box[0][1] - box[2][1])
        widthR = abs(box[0][0] - box[2][0])
 
        # remove the rectangles which are "too thin"(may be lines)
        if(heightR > widthR * 1.2):
            continue
 
        region.append(box)
        angle.append(theta)
        width.append(w)
        height.append(h)
        
    return region, angle , width, height


'''旋转图像并剪裁'''



def getTxetRegion(img,angle,region, width, height):
    
    imgH,imgW=img.shape[:2]
    RotImgMaxlength =np.int64( math.sqrt(imgH*imgH + imgW*imgW))
    if(width < height):    
        #box[2] is the top left point of the minimum outer rectangle and box[0] is the bottom rght point of the minimum outer rectangle
        M=cv2.getRotationMatrix2D((region[0][0], region[0][1]),(90+angle),1)
        rotated=cv2.warpAffine(img, M, (RotImgMaxlength, RotImgMaxlength))
        rotbox2 = pointRotation(img,90 + angle,region[2],region[0]) #xy is coordinates of box[2] after rotated
        bx2,by2=np.int64(rotbox2)
        #scaling get the final image to contain only the shark/fish.
        bw = np.int64(abs(bx2- region[0][0]))
        bh = np.int64(abs(by2- region[0][1]))
        textRegion = rotated[by2:by2+bh, bx2:bx2+bw]
        
    
    else:
       #box[2] is the top left point of the minimum outer rectangle and box[0] is the bottom rght point of the minimum outer rectangle
        M=cv2.getRotationMatrix2D((region[0][0], region[0][1]),angle,1)
        rotated=cv2.warpAffine(img, M, (RotImgMaxlength, RotImgMaxlength))
        rotbox2 = pointRotation(img,angle,region[2],region[0]) #xy is coordinates of box[2] after rotated
        rotbox1 = pointRotation(img,angle,region[1],region[0])
        bx2,by2=np.int64(rotbox2)
        bx1,by1=np.int64(rotbox1)
        #scaling get the final image to contain only the shark/fish.
        bw = np.int64(abs(bx2- region[0][0]))
        bh = np.int64(abs(by2- region[0][1]))
        textRegion = rotated[by1:by1+bh, bx1:bx1+bw]
    
    return textRegion

f = easygui.fileopenbox()
img = cv2.imread(f) 

testImage = img

cv2.imshow('image',testImage)
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows()


#Convert to YUV colourspace to exaggerate the fish/sea difference
grayImg = cv2.cvtColor(testImage, cv2.COLOR_RGB2GRAY)

cv2.imshow('gray',grayImg)
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows()

dilationImg = morphology(grayImg)
region , angle , width , height = findTextRegion(dilationImg)

textImage = []
for box, theta, w, h  in zip(region,angle,width,height):
    text_region = getTxetRegion(testImage,theta,box,w,h)
    textImage.append(text_region)
    
#     cv2.imshow("img", text_region)
#     cv2.waitKey(0) # waits until a key is pressed
#     cv2.destroyAllWindows()

Ordered_textImage =list(reversed(textImage))   

f = open("text.txt",'w+',encoding='utf-8')
f.truncate()
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
for text_img in Ordered_textImage:
    text_str = pytesseract.image_to_string(text_img)
    if(len(text_str)<2):
        continue
    else:
        f.write(text_str)
        f.write('\n')

f.seek(0,0)
for line in f:
    print(line)

f.close

cv2.destroyAllWindows()