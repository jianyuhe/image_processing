import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt


img = cv.imread("Shark.png")
# cv.imshow("before", img)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# split g,b,r
g = img[:,:,0]
b = img[:,:,1]
r = img[:,:,2]

# calculate hist
hist_r, bins_r = np.histogram(r, 256)
hist_g, bins_g = np.histogram(g, 256)
hist_b, bins_b = np.histogram(b, 256)

# calculate cdf
cdf_r = hist_r.cumsum()
cdf_g = hist_g.cumsum()
cdf_b = hist_b.cumsum()

# remap cdf to [0,255]
cdf_r = (cdf_r-cdf_r[0])*255/(cdf_r[-1]-1)
cdf_r = cdf_r.astype(np.uint8)# Transform from float64 back to unit8
cdf_g = (cdf_g-cdf_g[0])*255/(cdf_g[-1]-1)
cdf_g = cdf_g.astype(np.uint8)# Transform from float64 back to unit8
cdf_b = (cdf_b-cdf_b[0])*255/(cdf_b[-1]-1)
cdf_b = cdf_b.astype(np.uint8)# Transform from float64 back to unit8

# get pixel by cdf table
r2 = cdf_r[r]
g2 = cdf_g[g]
b2 = cdf_b[b]

# merge g,b,r channel
img2 = img.copy()
img2[:,:,0] = g2
img2[:,:,1] = b2
img2[:,:,2] = r2

# show img after histogram equalization
cv.imshow("img2", img2)



# alpha = 1.05 # Contrast control (1.0-3.0)
# beta = 10 # Brightness control (0-100)
# 
# adjusted = cv.convertScaleAbs(img2, alpha=alpha, beta=beta)
# cv.imshow('adjusted', adjusted)
# 
# adjusted = cv.cvtColor(adjusted, cv.COLOR_BGR2RGB)
# upper_blue = np.array([10, 185, 255])     ##[R value, G value, B value]
# lower_blue = np.array([10, 148, 232])
# mask = cv.inRange(adjusted, lower_blue, upper_blue)
# h,w,d = adjusted.shape
# print(adjusted.size)
# # 鱼的轮廓
# print(adjusted[401,706])
# print(adjusted[201,647])
# 
# masked_image = np.copy(img2)
# masked_image[mask == 255] = [255, 255, 255]
# cv.imshow('mask', masked_image)
# masked_image = cv.cvtColor(masked_image, cv.COLOR_BGR2RGB)
# plt.imshow(masked_image)
# plt.show()
cv.waitKey(0)
