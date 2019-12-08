#Image Processing Assignment1: to using python to find shark in occean
#author = Minhui Chen 
# DT228
# Student No: D17125347

#import package:
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui


#Reading Imgae
f = easygui.fileopenbox()
i = cv2.imread(f)

#convert BRG image to YUV
Yuv = cv2.cvtColor(i, cv2.COLOR_RGB2YUV)
y,u,v = cv2.split(Yuv)

# Values = y.ravel()
# plt.hist(Values, bins = 256, range = [0,256]);

Yequalize = cv2.equalizeHist(y)

#get new yuv image after equalize
MergeYUV = cv2.merge([Yequalize,u,v])

#covert the yuv to RGB
newImg = cv2.cvtColor(MergeYUV, cv2.COLOR_YUV2RGB)
cv2.imshow("newImg", newImg)
cv2.waitKey(0)

Vimge = cv2.add(v, 1)
cv2.imshow("Vimge", Vimge)
cv2.waitKey(0)

#using threshold to get ROI 
T = np.mean(Vimge)+np.std(Vimge)
T, ROI = cv2.threshold(u, thresh = T, maxval = 255, type = cv2.THRESH_BINARY_INV)
cv2.imshow("ROI", ROI)
cv2.waitKey(0)

kernel = np.ones((3,3), np.uint8)
img_bin = cv2.erode(ROI, kernel, iterations=1)
img_bin = cv2.dilate(img_bin, kernel, iterations=2)
cv2.imshow("img_bin", img_bin)
cv2.waitKey(0)

bitwiseNOT = cv2.bitwise_not(img_bin)

Shark = cv2.bitwise_and(i,i, mask = bitwiseNOT)
Shark[bitwiseNOT == 0] =[255,255,255]
Shark = cv2.GaussianBlur(Shark, (3,3), 0)

cv2.imshow("Shark", Shark)
cv2.waitKey(0)



cv2.waitKey(0)


# =============================================================================
# k = py.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=float)
# f = cv2.filter2D(S1,ddepth=-1, kernel = k)
# =============================================================================