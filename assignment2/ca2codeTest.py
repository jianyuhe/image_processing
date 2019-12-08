# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 02:33:35 2019

@author: Yuweichen
"""


from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import easygui
import math

#open image by easygui
f = easygui.fileopenbox()
img = cv2.imread(f) 



grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    #get binary image

cv2.imshow("test",grayImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
  
def whiteBallRegion(center,radius):
    regionArea = []
    checkRegion = img[center[1] - radius:center[1]+radius , center[0] - radius:center[0]+radius]
    low_hsv = np.array([105,105,105])
    high_hsv = np.array([255,255,255])
    
    mask = cv2.inRange(checkRegion,lowerb=low_hsv,upperb=high_hsv)
#    cv2.imshow("test",mask)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        regionArea.append(area)
    if(len(regionArea)!=0 ):
        return  max(regionArea)/(3.14*pow(radius,2))
    else:
        return 0
 



tvalue =np.mean(grayImg)+np.std(grayImg)
(t, maskLayer) = cv2.threshold(src =grayImg, 
    thresh = tvalue, 
    maxval = 255, 
    type = cv2.THRESH_BINARY_INV)

#low_hsv = np.array([0,0,46])
#high_hsv = np.array([180,43,255])
#mask = cv2.inRange(hsv,lowerb=low_hsv,upperb=high_hsv)
#cv2.imshow("test",maskLayer)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#Remove spots around fish by iteratively repeating morphological operations
OldMask = maskLayer
shape = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))


NewMask= cv2.morphologyEx(OldMask,cv2.MORPH_OPEN,shape)

NewMask= cv2.dilate(NewMask,shape)
NewMask= cv2.dilate(NewMask,shape)
NewMask= cv2.erode(NewMask,shape)



contours, hierarchy = cv2.findContours(NewMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow("test",NewMask)
cv2.waitKey(0)
cv2.destroyAllWindows()

ballc=(0,0)
ballr=0
ballp=0
#for loop traverse contours and remove all small area of the contourns(small area of the contourns may be spots)
for i in range(len(contours)):
    cnt = contours[i]
    area = cv2.contourArea(cnt) 
        
    if(area < 500):
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
    if(heightR > widthR * 3):
       continue
   
    if(widthR > heightR * 3):
       continue
   
       
    (x, y), radius = cv2.minEnclosingCircle(cnt)
         
    center = (int(x), int(y))
    radius = int(radius)
    
    #cv2.circle(img, center, radius, (255, 0, 0), 2)
    r= np.int64(radius*0.9)
    
    p=whiteBallRegion(center,r)

    
    if(p>ballp):
        ballp = p
        ballc = center
        ballr = radius
        
    print(ballp,ballc,ballr)

h,w,d=img.shape

low_hsv = np.array([105,105,105])
high_hsv = np.array([255,255,255])
imgcopy = img



mask1=np.zeros_like(OldMask)
mask2=np.zeros_like(OldMask)
mask1 = cv2.rectangle(mask1, (ballc[0]-ballr,ballc[1]-ballr), (ballc[0]+ballr,ballc[1]+ballr), (255,255,255), 5)
mask2=cv2.circle(mask2, ballc, ballr+2, (255, 255, 255), -1)
dst_NS = cv2.inpaint(img,mask2,5,cv2.INPAINT_NS)
imgcopy[ballc[1] - ballr:ballc[1]+ballr , ballc[0] - ballr:ballc[0]+ballr] = imgcopy[ballc[1] - ballr:ballc[1]+ballr , ballc[0] + ballr:ballc[0]+ballr+ballr+ballr]
dst_TELEA = cv2.inpaint(imgcopy,mask1,3,cv2.INPAINT_TELEA)



cv2.imshow("mask1",dst_TELEA)
cv2.imshow("ns2",imgcopy)
cv2.imshow("mask2",dst_NS)
cv2.waitKey(0)
cv2.destroyAllWindows()