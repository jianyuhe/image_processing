import cv2
import numpy as np 
from matplotlib import pyplot as plt


img = cv2.imread("Shark.png")

# cv2.imshow("before", img)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
vmin = v.min()
vmax = v.max()
for i in range(0, len(v)):
   v[i] = 255* ((v[i] - vmin)/(vmax - vmin))

nhsv = cv2.merge([h,s,v])
img2 = cv2.cvtColor(nhsv, cv2.COLOR_HSV2BGR)


cv2.imshow("img2", img2)

def nothing(x):
    pass

cv2.namedWindow('window')
cv2.createTrackbar('alpha','window',0,3,nothing)
cv2.createTrackbar('beta','window',0,100,nothing)
cv2.createTrackbar('threshlow','window',0,255,nothing)
cv2.createTrackbar('threshhigh','window',0,255,nothing)
cv2.createTrackbar('yuv_vlow','window',0,255,nothing)
cv2.createTrackbar('yuv_vhigh','window',0,255,nothing)

cv2.setTrackbarPos('alpha','window',1)
cv2.setTrackbarPos('beta','window',10)
# alpha = 1.5 # Contrast control (1.0-3.0)
# beta = 0 # Brightness control (0-100)


while(1):
   alpha=cv2.getTrackbarPos('alpha','window')
   beta=cv2.getTrackbarPos('beta','window')
   adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
   h,w,d = adjusted.shape
   cv2.imshow('adjusted', adjusted)
   
   
   maskimg = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
   
# lower_blue = np.array([100,43,46])     ##[R value, G value, B value]
# upper_blue = np.array([110,240,240])
# mask = cv2.inRange(adjusted, lower_blue, upper_blue)
   y,u,v = cv2.split(maskimg)
#    y1,u1,v1 = cv2.split(img2)
   low=cv2.getTrackbarPos('threshlow','window')
   high=cv2.getTrackbarPos('threshhigh','window')
   vlow=cv2.getTrackbarPos('yuv_vlow','window')
   vhigh=cv2.getTrackbarPos('yuv_vhigh','window')
   ret,thresh1 = cv2.threshold(u,low,high,cv2.THRESH_BINARY_INV)
#    ret,thresh2 = cv2.threshold(v1,vlow,vhigh,cv2.THRESH_BINARY_INV)
   cv2.imshow('mask', thresh1)
   masked_image = np.copy(img2)
   masked_image[thresh1 == 0] = [255, 255, 255]
#    masked_image[thresh2 != 0] = [255, 255, 255]
   cv2.imshow('mask1', masked_image)
   
   k=cv2.waitKey(1)
   k = cv2.waitKey(1) & 0xFF
   if k == ord('m'):
     mode = not mode
   elif k == 27:
     break



print(h, w, d)
print(adjusted.size)
# 鱼的轮廓
print(adjusted[h-10,0])
print(adjusted[0,w-10])
print(adjusted[208,663])

# masked_image = np.copy(img2)
# masked_image[mask == 255] = [255, 255, 255]
# cv2.imshow('mask', masked_image)
# masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
# plt.imshow(masked_image)
# plt.show()
