import cv2 
from matplotlib import pyplot as plt
img = cv2.imread('rose.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h,w,d = img.shape
# S = cv2.resize(img, dsize=(2*w, 2*h))
w1 = w//2
h1 = h//2
C =img[0:w1,0:h1]
h2,w2,d2 = C.shape
M =cv2.getRotationMatrix2D(center=(w2//2,h2//2),angle=45, scale=1)
R =cv2.warpAffine(C, M = M,dsize=(w2,h2))
plt.subplot(2, 1, 1)
plt.imshow(C)   

plt.subplot(2, 1, 2)
plt.imshow(R)
plt.show()
key = cv2.waitKey(0)