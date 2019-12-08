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

'''旋转图像并剪裁'''
def rotate(
        img,  # 图片
        pt1, pt2, pt3, pt4 ):
    
    print (pt1,pt2,pt3,pt4)
    withRect = math.sqrt((pt4[0] - pt1[0]) ** 2 + (pt4[1] - pt1[1]) ** 2)  # 矩形框的宽度
    heightRect = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) **2)
    print (withRect,heightRect)
    angle = math.acos((pt4[0] - pt1[0]) / withRect) * (180 / math.pi)  # 矩形框旋转角度
    print (angle)
 
    if pt4[1] > pt1[1]:
        print ("顺时针旋转")
    else:
        print ("逆时针旋转")
        angle = -angle
 
    height = img.shape[0]  # 原始图像高度
    width = img.shape[1]   # 原始图像宽度
    
    rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)  # 按angle角度旋转图像
    heightNew = int(width * math.fabs(math.sin(math.radians(angle))) + height * math.fabs(math.cos(math.radians(angle))))
    widthNew = int(height * math.fabs(math.sin(math.radians(angle))) + width * math.fabs(math.cos(math.radians(angle))))
 
    rotateMat[0, 2] += (widthNew - width) / 2
    rotateMat[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, rotateMat, (widthNew, heightNew), borderValue=(255, 255, 255))
 
    # 旋转后图像的四点坐标
    [[pt1[0]], [pt1[1]]] = np.dot(rotateMat, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(rotateMat, np.array([[pt3[0]], [pt3[1]], [1]]))
    [[pt2[0]], [pt2[1]]] = np.dot(rotateMat, np.array([[pt2[0]], [pt2[1]], [1]]))
    [[pt4[0]], [pt4[1]]] = np.dot(rotateMat, np.array([[pt4[0]], [pt4[1]], [1]]))
 
    # 处理反转的情况
    if pt2[1] > pt4[1]:
        pt2[1],pt4[1] = pt4[1],pt2[1]
    if pt1[0] > pt3[0]:
        pt1[0],pt3[0] = pt3[0],pt1[0]
 
    imgOut = imgRotation[int(pt2[1]):int(pt4[1]), int(pt1[0]):int(pt3[0])]
#    cv2.imwrite(newImagePath, imgOut)  # 裁减得到的旋转矩形框
    return imgOut  # rotated image


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
    
    cv2.imshow("img", text_region)
    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows()
    
#for imgtext in textImage:
#    tvalue =np.mean(imgtext)+np.std(imgtext)
#    (t, textImgMask) = cv2.threshold(src =imgtext, thresh = tvalue, maxval = 255, type = cv2.THRESH_BINARY_INV)
#    region , angle , width , height = findTextRegion(imgtext)
#    for box, theta, w, h  in zip(region,angle,width,height):
#        imgH,imgW=imgtext.shape[:2]
#        if(w < h):  
#            M=cv2.getRotationMatrix2D((box[0][0], box[0][1]),(90+theta),1.0)
#            rotated=cv2.warpAffine(imgtext, M, (imgH*2, imgW*2))
#            xy = Nrotate(imgtext,90 + theta,box[2],box[0]) #xy is coordinates of box[2] after rotated
#            bx,by=np.int64(xy)
#            #scaling get the final image to contain only the shark/fish.
#            bw = np.int64(abs(bx- box[0][0]))
#            bh = np.int64(abs(by- box[0][1]))
#           
#    #        cv2.imshow("imgOut.jpg",textRegion)
#    #        cv2.waitKey(0) # waits until a key is pressed
#    #        cv2.destroyAllWindows()
#
#        else: 
#            M=cv2.getRotationMatrix2D((box[0][0], box[0][1]),theta,1.0)
#            rotated=cv2.warpAffine(imgtext, M, (imgH*4, imgW*4))
#            xy = Nrotate(imgtext,theta,box[2],box[0]) #xy is coordinates of box[2] after rotated
#            box1 = Nrotate(imgtext,theta,box[1],box[0])
#            bx,by=np.int64(xy)
#            bx1,by1=np.int64(box1)
#            #scaling get the final image to contain only the shark/fish.
#            bw = np.int64(abs(bx- box[0][0]))
#            bh = np.int64(abs(by- box[0][1]))
#           
#    #        cv2.imshow("imgOut.jpg",textRegion)
#    #        cv2.waitKey(0) # waits until a key is pressed
#    #        cv2.destroyAllWindows()
#           
#       
#        cv2.drawContours(imgtext, [box], 0, (0,0, 0), 2)
##
#        cv2.imshow("imgOut.jpg",imgtext)
#        cv2.waitKey(0) # waits until a key is pressed
#        cv2.destroyAllWindows()
for text_img in textImage:
    
    gray_text_img = cv2.cvtColor(text_img, cv2.COLOR_RGB2GRAY)
    mser = cv2.MSER_create()
    
    ret, thresh = cv2.threshold(gray_text_img, 10, 255, cv2.THRESH_OTSU)
    
  
    
    shape = cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))
    
    
    threshMask = cv2.dilate(thresh,shape)
  
    h,w = thresh.shape[:2] 
    b_thresh = cv2.resize(thresh,(w*2,h*2))
    b_thresh = cv2.dilate(b_thresh,shape)
    b_text_img = cv2.resize(text_img,(w*2,h*2))
    nh_copy = b_thresh
    img_copy = b_text_img
    
    #letter_image = []
    # With the rects you can e.g. crop the letters
#    regions, rects = mser.detectRegions(b_thresh)
#    letter_region , letter_angle , letter_width , letter_height = findTextRegion(b_thresh)
#   
# 
#    
#    for i in range(len(rects)):
#        x = rects[i][0]
#        y = rects[i][1]
#        w = rects[i][2]
#        h = rects[i][3]
#        x1 = rects[i-1][0]
#        y1 = rects[i-1][1]
#        w1 = rects[i-1][2]
#        h1 = rects[i-1][3]
#        cv2.rectangle(b_text_img, (x, y), (x+w, y+h), color=(0, 0, 255), thickness=1)
#       # cv2.rectangle(img, (x1+w1,y1), (x,y+h1), color=(0, 255, 255), thickness=1)
#        print(x-rects[i-1][0])
#        
    
    cv2.imshow('thresh',b_thresh)
    cv2.imshow('sd',b_text_img)
    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows()  
    
#def nms(bounding_boxes, confidence_score, threshold):
#    # If no bounding boxes, return empty list
#    if len(bounding_boxes) == 0:
#        return [], []
#
#    # Bounding boxes
#    boxes = np.array(bounding_boxes)
#
#    # coordinates of bounding boxes
#    start_x = boxes[:, 0]
#    start_y = boxes[:, 1]
#    end_x = boxes[:, 2]
#    end_y = boxes[:, 3]
#
#    # Confidence scores of bounding boxes
#    score = np.array(confidence_score)
#
#    # Picked bounding boxes
#    picked_boxes = []
#    picked_score = []
#
#    # Compute areas of bounding boxes
#    areas = (end_x - start_x + 1) * (end_y - start_y + 1)
#
#    # Sort by confidence score of bounding boxes
#    order = np.argsort(score)
#
#    # Iterate bounding boxes
#    while order.size > 0:
#        # The index of largest confidence score
#        index = order[-1]
#
#        # Pick the bounding box with largest confidence score
#        picked_boxes.append(bounding_boxes[index])
#        picked_score.append(confidence_score[index])
#        a=start_x[index]
#        b=order[:-1]
#        c=start_x[order[:-1]]
#        # Compute ordinates of intersection-over-union(IOU)
#        x1 = np.maximum(start_x[index], start_x[order[:-1]])
#        x2 = np.minimum(end_x[index], end_x[order[:-1]])
#        y1 = np.maximum(start_y[index], start_y[order[:-1]])
#        y2 = np.minimum(end_y[index], end_y[order[:-1]])
#
#        # Compute areas of intersection-over-union
#        w = np.maximum(0.0, x2 - x1 + 1)
#        h = np.maximum(0.0, y2 - y1 + 1)
#        intersection = w * h
#
#        # Compute the ratio between intersection and union
#        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)
#
#        left = np.where(ratio < threshold)
#        order = order[left]
#
#    return picked_boxes, picked_score
#
#
#
#
## Bounding boxes
#bounding_boxes = rects[:3]
#confidence_score = [0.9, 0.75, 0.8]
#
## Read image
#image = img_copy
#
## Copy image as original
#org = image.copy()
#
## Draw parameters
#font = cv2.FONT_HERSHEY_SIMPLEX
#font_scale = 1
#thickness = 2
#
## IoU threshold
#threshold = 0.4
#
## Draw bounding boxes and confidence score
#for (start_x, start_y, end_x, end_y), confidence in zip(bounding_boxes, confidence_score):
#    #(w, h), baseline = cv2.getTextSize(str(confidence), font, font_scale, thickness)
#    cv2.rectangle(org, (start_x, start_y - (2 * baseline + 5)), (start_x + w, start_y), (0, 255, 255), -1)
#    cv2.rectangle(org, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
#    #cv2.putText(org, str(confidence), (start_x, start_y), font, font_scale, (0, 0, 0), thickness)
#
## Run non-max suppression algorithm
#picked_boxes, picked_score = nms(bounding_boxes, confidence_score, threshold)
#
## Draw bounding boxes and confidence score after non-maximum supression
#for (start_x, start_y, end_x, end_y), confidence in zip(picked_boxes, picked_score):
#    (w, h), baseline = cv2.getTextSize(str(confidence), font, font_scale, thickness)
#    cv2.rectangle(image, (start_x, start_y - (2 * baseline + 5)), (start_x + w, start_y), (0, 255, 255), -1)
#    cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
#    cv2.putText(image, str(confidence), (start_x, start_y), font, font_scale, (0, 0, 0), thickness)
#
## Show image
#cv2.imshow('Original', org)
#cv2.imshow('NMS', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#cv2.imshow("img",b_thresh)
#cv2.waitKey(0) # waits until a key is pressed
#cv2.destroyAllWindows()
#
def non_max_suppression_slow(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
 
	# initialize the list of picked indexes
	pick = []
 
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
 
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
    	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list, add the index
		# value to the list of picked indexes, then initialize
		# the suppression list (i.e. indexes that will be deleted)
		# using the last index
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		suppress = [last]
        
     
        # loop over all indexes in the indexes list
		for pos in range(0, last):
			# grab the current index
			j = idxs[pos]
 
			# find the largest (x, y) coordinates for the start of
			# the bounding box and the smallest (x, y) coordinates
			# for the end of the bounding box
			xx1 = max(x1[i], x1[j])
			yy1 = max(y1[i], y1[j])
			xx2 = min(x2[i], x2[j])
			yy2 = min(y2[i], y2[j])
 
			# compute the width and height of the bounding box
			w = max(0, xx2 - xx1 + 1)
			h = max(0, yy2 - yy1 + 1)
 
			# compute the ratio of overlap between the computed
			# bounding box and the bounding box in the area list
			overlap = float(w * h) / area[j]
 
			# if there is sufficient overlap, suppress the
			# current bounding box
			if overlap > overlapThresh:
				suppress.append(pos)
 
		# delete all indexes from the index list that are in the
		# suppression list
		idxs = np.delete(idxs, suppress)
 
	# return only the bounding boxes that were picked
	return boxes[pick]

boundingBoxes = rects
# loop over the bounding boxes for each image and draw them
orig = img_copy
image = img_copy
for (startX, startY, endX, endY) in boundingBoxes:
	cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)
 
	# perform non-maximum suppression on the bounding boxes
pick = non_max_suppression_slow(boundingBoxes, 0.3)
	#print "[x] after applying non-maximum, %d bounding boxes" % (len(pick))
 
	# loop over the picked bounding boxes and draw them
for (startX, startY, endX, endY) in pick:
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
 
	# display the images
cv2.imshow("Original", img_copy)
cv2.imshow("After NMS", image)
cv2.waitKey(0)
cv2.destroyAllWindows()