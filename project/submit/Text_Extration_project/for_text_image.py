# -*- coding: utf-8 -*-
"""
Created on  Nov 1 14:00:30 2019

Project Title: Text extraction.
Project Objective: Extract text in images
Team Members: Yuwei Chen, Minhui Chen, Jianyu He


"""
'''
Progrm description
            This program will extract text from text image (like screen capture from what's up,screen capture from text document .ie).
            For some nature images we have another program
            There are 5 steps to achieve print text which from image on screen.
            Step1:Extract text region by morphology method and threshold function. (usually one sentence as one text region)
            Step2:For some tilted text regions, turn them back.
            Step3:Get text string from turned text regions by pytesseract.image_to_string( )
            Step4:Store text into text file
            Step5:Read text from text file and print them on screen 
'''
'''
For Innovation In This Program:
   We are not totally rely on pytesseract.image_to_string( ) function, 
   if the whole original image as the argyment of pytesseract.image_to_string( ) function,
   we can get some text, but because of some noise interference, the effect is not so good. 
   So in order to make this project more efficient,
   we get the text region fist then use pytesseract.image_to_string( ) function to get text,
   the result is much better. And we write a algorithm which
   based on morphology method and threshold function by ourselfs to get text region
   instead of relying on text detection from website.
   In order to perform  pytesseract.image_to_string( ) function better, 
   we write a algorithm to turn text regions which are tilted back.
    
       
'''
import pytesseract
import cv2
import numpy as np
import easygui
import math
from math import *
from matplotlib import pyplot as plt


#morphologe function to get a binary image without nosie(or less noise)
def morphology(grayImg):
    # apply av horizontal Sobel filter
    sobel = cv2.Sobel(grayImg, cv2.CV_8U, 1, 0, ksize = 3)
    
    #get binary image
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
 
    #Remove spots noise by apply morphological operations
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))
    dilation = cv2.dilate(binary, element, iterations = 1)

    return  dilation 

#function Point A goes around point B angle degrees counterclockwise in image
#it's different from normall point rotation becaurse the origin in the image is different from the usual one.
def pointRotation(image,angle,pointA,pointB):   
  h,w = image.shape[:2]
  ax = pointA[0]
  ay = h - pointA[1]
  bx = pointB[0]
  by = h - pointB[1]
  
  #Trigonometric principle
  nRotatex = (ax-bx)*math.cos(math.pi / 180.0 * angle) - (ay-by)*math.sin(math.pi / 180.0 * angle) + bx
  nRotatey = (ax-bx)*math.sin(math.pi / 180.0 * angle) + (ay-by)*math.cos(math.pi / 180.0 * angle) + by
  
  #The coordinates after the point is rotated in image
  rotatedPoint =(nRotatex, h - nRotatey)
  
  return rotatedPoint

#findTextRegion function to find the text regions in image, so we can crop these region in order to extrac text better
def findTextRegion(grayImg):
    region = [] #list for text region
    angle = [] #Rotation angle of text region
    width=[] #width of text region 
    height = [] #height of text region
    
    #find contours of regions from gray image of original image
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
        
        #append values into lists
        region.append(box)
        angle.append(theta)
        width.append(w)
        height.append(h)
        
    return region, angle , width, height


#this function is to crop text region after finding text region
#every text region returned from here are turned back(without rotation)
def getTxetRegion(img,angle,region, width, height):
    #To prevent text region from exceeding the picture when rotating(if exceed we cant get the whole text region )
    #we need resize picture, the max size should be( math.sqrt(imgH*imgH + imgW*imgW), math.sqrt(imgH*imgH + imgW*imgW))
    imgH,imgW=img.shape[:2]
    RotImgMaxlength =np.int64( math.sqrt(imgH*imgH + imgW*imgW))
    
    #There are two options here, if width < height, we need anticlockwise rotation
    #otherwise clockwise rotation
    
    #for anticlockwise rotation
    if(width < height):    
        #box[2] is the top left point of the minimum outer rectangle and box[0] is the bottom rght point of the minimum outer rectangle
        M=cv2.getRotationMatrix2D((region[0][0], region[0][1]),(90+angle),1)
        rotated=cv2.warpAffine(img, M, (RotImgMaxlength, RotImgMaxlength))
        rotbox2 = pointRotation(img,90 + angle,region[2],region[0]) #rotbox2 is coordinates of box[2] after rotated
        bx2,by2=np.int64(rotbox2)
        
        #scaling get the text region from rotated image, so the text region here should without rotation
        bw = np.int64(abs(bx2- region[0][0]))
        bh = np.int64(abs(by2- region[0][1]))
        textRegion = rotated[by2:by2+bh, bx2:bx2+bw]
          
    #for clockwise rotation
    else:
        M=cv2.getRotationMatrix2D((region[0][0], region[0][1]),angle,1)
        rotated=cv2.warpAffine(img, M, (RotImgMaxlength, RotImgMaxlength))
        rotbox2 = pointRotation(img,angle,region[2],region[0]) #rotbox2 is coordinates of box[2] after rotated
        rotbox1 = pointRotation(img,angle,region[1],region[0]) ##rotbox1 is coordinates of box[1] after rotated
        bx2,by2=np.int64(rotbox2)
        bx1,by1=np.int64(rotbox1)
        
       #scaling get the text region from rotated image, so the text region here should without rotation
        bw = np.int64(abs(bx2- region[0][0]))
        bh = np.int64(abs(by2- region[0][1]))
        textRegion = rotated[by1:by1+bh, bx1:bx1+bw]
    
    return textRegion

#open image by easygui.fileopenbox()
f = easygui.fileopenbox()
image = cv2.imread(f) 

#Convert to gray image
grayImg = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#get a binary image without nosie(or less noise) by morphology( )  function
dilationImg = morphology(grayImg)

#get region , angle , width , height of text region
region , angle , width , height = findTextRegion(dilationImg)

#store text regions into list
textImage = []
for box, theta, w, h  in zip(region,angle,width,height):
    text_region = getTxetRegion(image,theta,box,w,h)
    textImage.append(text_region)

#the last element of list should be the first text region 
Ordered_textImage =list(reversed(textImage))   

#open text file
#create if not exist
text_file = open("text.txt",'w+',encoding='utf-8')

#Clear file content
text_file.truncate()

#store text string into text file
for text_img in Ordered_textImage:
    text_str = pytesseract.image_to_string(text_img)
    if(len(text_str)<2): #remove empty text string
        continue
    else:
        text_file.write(text_str)
        text_file.write('\n')

#read text begin with first line
text_file.seek(0,0)
#print text on screen
for line in text_file:
    print(line)

#close text file
text_file.close

#show original
cv2.imshow('image',image)
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows()
