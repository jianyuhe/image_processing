import cv2

from matplotlib import pyplot as plt
import numpy as np
src = cv2.imread("fish.png")  

hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
vmin = v.min()
vmax = v.max()
for i in range(0, len(v)):
   v[i] = 255* ((v[i] - vmin)/(vmax - vmin))

nimg = cv2.merge([h,s,v])
out = cv2.cvtColor(nimg, cv2.COLOR_HSV2BGR)
cv2.imshow("input", hsv)
cv2.imshow("output", out)

cv2.waitKey(0)
cv2.destroyAllWindows()

