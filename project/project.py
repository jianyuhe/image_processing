import cv2
import numpy as np
import easygui
import math
from matplotlib import pyplot as plt
from matplotlib import image as image

'''
The original idea is first rect each letters from whatsapp image,
then type a-z, A-Z and 0-9 on whatsapp app, after crop as a image,
then rect each letters and numbers use to create image of each rect
save into template folder, after we compare histogram of both rect
one from whatsapp other one from template, if both histogram
of rect is same, that we can knows what letter of the rect.
final we can print all letters from whatsapp image without space.
for create space for letters: we know position of each rect,
then we can find length of space is more longer with lenght of each letters.
that we can know position of space. we can create space for each letters.
'''

# read whatsapp image
img = cv2.imread('WhatsApp.png',0)
# read template imgae
let = cv2.imread('template\\32.png',0)

(h, w) = img.shape[:2]
image_size = h*w
# create mser
mser = cv2.MSER_create()
# create mask image using whatsapp image
ret, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_OTSU)
# find rect of each letters from whatsapp image
regions, rects = mser.detectRegions(thresh)


# use loop to find each rect of whatsapp image
for i in range(len(rects)):
    x = rects[i][0]
    y = rects[i][1]
    w = rects[i][2]
    h = rects[i][3]
    cv2.rectangle(img, (x, y), (x+w, y+h), color=(0, 0, 255), thickness=1)
#     change some size of rect from whatsapp image for template image
    resized = cv2.resize(let, (w,h), interpolation = cv2.INTER_AREA)
#     create histogram of rect from whatsapp image
    hist1 = cv2.calcHist([img[y:y+h, x:x+w]],[0],None,[256],[0,256])
#     create histogram of template image
    hist2 = cv2.calcHist([resized],[0],None,[256],[0,256])
#     compare both histogram
    corr = cv2.compareHist(hist1, hist2, 0)
#     print(corr)
# #     cv2.imshow('sd',img[y:y+h, x:x+w])
#     cv2.imshow('sds',resized)
#     cv2.waitKey(0)
    
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.subplot(2, 1, 1)
    plt.title('hist from whatsapp image')
    plt.plot(hist1)
    plt.subplot(2, 1, 2)
    plt.title('hist from letter image')
    plt.plot(hist2)
    plt.xlim([0,256])
    plt.show()



cv2.imshow('sd',img)
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows()