import cv2 
from matplotlib import pyplot as plt 
img = cv2.imread('wartime.jpg',0) 
H = cv2.equalizeHist(img)
cv2.imshow("war", img)
cv2.imshow("new", H)
# alternative way to find histogram of an image 
plt.subplot(2, 1, 1)
plt.hist(img.ravel(),256,[0,256])   

plt.subplot(2, 1, 2)
plt.hist(H.ravel(),256,[0,256]) 
plt.show()  