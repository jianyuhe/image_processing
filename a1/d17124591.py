import cv2
import numpy as np 
from matplotlib import pyplot as plt
from math import *
import random
import easygui
from time import perf_counter
import time
f = easygui.fileopenbox()
img = cv2.imread(f)
#Convert to an appropriate colourspace to exaggerate the fish/sea difference;
#use HSV color space and equalizeHist function to exaggerate the fish/sea difference;
origin = np.copy(img)
origin = cv2.cvtColor(origin, cv2.COLOR_BGR2RGB)
plt.subplot(2,3,1) #create 2x3 table to store image 
plt.title("original image")
plt.imshow(origin)


hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #change bgr color space to hsv
h, s, v = cv2.split(hsv)  #splite h,s,v chanal from hsv
vmin = v.min() #find minimum intensity from origin image
vmax = v.max() #find maximum intensity from origin image
for i in range(0, len(v)):     #use formula from notes, this formula use for find new intensity can be equalize image
   v[i] = 255* ((v[i] - vmin)/(vmax - vmin))

nhsv = cv2.merge([h,s,v]) #merge new intensity 
img2 = cv2.cvtColor(nhsv, cv2.COLOR_HSV2BGR) 

img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
plt.subplot(2,3,2)
plt.title("equalize image")
plt.imshow(img2)

# Enhance the images to increase contrast and definition;
alpha = 1.5 # Contrast control (1.0-3.0)
beta = 10 # Brightness control (0-100)
img2 = cv2.convertScaleAbs(img2, alpha=alpha, beta=beta) #use this function to chage contrast and brightness of image


plt.subplot(2,3,3)
plt.title("enhanced image")
plt.imshow(img2)





# Extract the fish from the images and convert the background to white;
# use u channal of YUV color space for threshold
maskimg = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

y,u,v = cv2.split(maskimg) #splite y,u,v chanal from YUV color space

#find a number can use for select better threshold base on statistical analysis 
nv = cv2.add(u, -6)
tvalue = np.mean(nv) + np.std(nv)
ret,thresh1 = cv2.threshold(u,tvalue,255,cv2.THRESH_BINARY_INV) #use threshold to clear sea
# use Erosion to clear small blue point
kernel = np.ones((3,3), np.uint8)
erorsion_img = cv2.erode(thresh1, kernel, iterations=3)

# use dilate to fix missing space from fish
dilate_img = cv2.dilate(erorsion_img, kernel, iterations=6)

masked_image = np.copy(img2)
masked_image[dilate_img == 0] = [255, 255, 255] #cover background without fish to white

option = np.copy(masked_image)

plt.subplot(2,3,4)
plt.title("fish with out see")
plt.imshow(masked_image)

#extract fish use findcontour function, this function can use rect to contain each object in threshold image
contours, hier = cv2.findContours(dilate_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
x =0
y =0
wid =0
hei =0
theta1 =0
#find x, y , width, height and theta of rect
for cidx,cnt in enumerate(contours):
    (cx, cy, width, height) = cv2.boundingRect(cnt) #find detail of rect
    minAreaRect = cv2.minAreaRect(cnt)
    rectCnt = np.int64(cv2.boxPoints(minAreaRect))
    cv2.drawContours(option, [rectCnt], 0, (0,255,0), 3) #show rect in the image
  
    
    ((mcx, mcy), (mwidth, mheight), mtheta) = cv2.minAreaRect(cnt)  #find details of minimun area of rect
    rectCnt = np.int64(cv2.boxPoints(minAreaRect))
    
  
    if(mtheta%15 != 0):      #check which one is rect of fish, because i compare fish image and shark image to find the similar
        wid = np.int(width)
        hei = np.int(height)
        x = np.int(cx)
        y = np.int(cy)
        theta1 = mtheta
      

fishcut = masked_image[y:y+hei, x:x+wid] #extrct fish

plt.subplot(2,3,5)
plt.title("extract fish")
plt.imshow(fishcut)

#Enhance the fish portion of the image;
# GaussianBlur
fishcut = cv2.GaussianBlur(fishcut, (5,5), 1.5);


# create a blank image then set fish in the white background of image
blank = np.zeros((1000,1000,3), np.uint8)  #create a blank image
blank.fill(255)     #fill color of new image to white 
x_offset=y_offset=1000//6
blank[y_offset:y_offset+fishcut.shape[0], x_offset:x_offset+fishcut.shape[1]] = fishcut  #set fish image into new blank image

plt.subplot(2,3,6)
plt.title("cover fish to blank image")
plt.imshow(blank)
plt.show()

#Optional: automatically crop and rotate the image to contain only the shark/fish.

degree = 0
count = 1
while(degree < 360):  #if rotate 360 then stop
    
    times = perf_counter() #create time counter
   
    if(times >= count):
        count = count + 1
        degree = degree + 10
       
    h, w, c = fishcut.shape
    heightNew=int(width*fabs(sin(radians(degree)))+h*fabs(cos(radians(degree)))) #create new height is depend height of fish
    widthNew=int(height*fabs(sin(radians(degree)))+w*fabs(cos(radians(degree)))) #create new width of window is depend width of fish
    matRotation=cv2.getRotationMatrix2D((w/2,h/2),degree,1) 
#set rotate position for fish
    matRotation[0,2] +=(widthNew-w)/2
    matRotation[1,2] +=(heightNew-h)/2
    imgRotation=cv2.warpAffine(fishcut,matRotation,(widthNew,heightNew),borderValue=(255,255,255)) # create new image which image rotate already
    imgRotation = cv2.cvtColor(imgRotation, cv2.COLOR_RGB2BGR) #change color space
    cv2.imshow('fish rotate', imgRotation)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    time.sleep(1) #stop while loop 1 seconds
    

cv2.waitKey()
cv2.destroyAllWindows()




