# Import necessary packages

import math
import numpy as np
from cv2 import cv2
import easygui
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

'''
    Program Name: Shark Attack - Assignment 1
    Studnet Name: Chenxi Zhang
    Student Number: C16434996
    Description

        Using opencv techniques to sucessfully crop the shark from the sea.

        First, the image is converted to HSV colorspace to prepare for increasing contrast/intensity.
        The color channel using is the V channel, which is the value/intensity of the image.
        
        Then, by using CLAHE, adaptive histogram equalization to soften the contrast into reasonable range 
        and giving better result. (values I tried worked the best by examining the outcome)

        The image is thresholded using best practice mean +/- standard deviation of the image, binary mask is generated.
        From using morphology open, we exclude some white dots and fill the blanks inside the shark from the binary image to get better results

        Using the binary mask to crop the shark from the background, then by whiten out a same sized matrix of the image,
        place the cropped shark onto the whiten canvas and use cv2.add() the original image to get the shark with white background.

        After the shark has cropped to a white background, apply CLAHE again to increase the contrast of the shark from the sea water.

        By finding the minimum area rectangle(minAreaRect()) that surrond the shark, we use the co-ordinate provided by that function to slice the shark from
        white background.

        Create a canvas that is twice as big of the sliced shark and transform the sliced image onto the center of the canvas, this will prevent
        the shark from falling off the edge when rotating the image. A rotation matrix is made and use the angle given by minAreaRect() we can
        rotate the fish back to horizontal angle.

    Date: 21/10/19
'''


# Histogram equalization algorithm.
# Not using this because found adaptive histogram equaliztion

# def Equalize(channel):
#     ch = channel.copy()

#     ch_min = ch.min()
#     ch_max = ch.max()

#     for i in range(0, len(ch)):
#         ch[i] = 255 * ((ch[i] - ch_min) / (ch_max - ch_min))

#     return ch


# Function that returns the largest contour details
# Loops through the contours and find the area of the contour by
# the function contours ares
# @return: largest contour from contours
def findLargestContours(c):

    largest = c[0]

    # Find largest contour from contours by check the
    # size of the rectangle drawn (w*h)
    for c in contours:
        size = cv2.contourArea(c)

        if size > cv2.contourArea(largest):
            largest = c

    return largest

# Load image from project folder
image_file = easygui.fileopenbox()

# Read image from file
img = cv2.imread(image_file)


# Convert the image into appropiate colorspace for contrast
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Work with the V channel, the intensity channel.
h, s, v = cv2.split(hsv)


plt.figure()
plt.subplot(1,2,1)
plt.title('Original Grayscale')
plt.imshow(v, cmap='gray')

# Create a adaptive histogram equalization variable
clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(4, 4))
# Remove noise from original grayscale image
G = cv2.medianBlur(v, 5)
# Apply equalization
G = clahe.apply(G)

# Show equalized image
plt.subplot(1,2,2)
plt.title('After CLAHE algorithm')
plt.imshow(G, cmap='gray')
# cv2.imshow('CLAHE', G)

# Find the threshold of the image, using best practice mean +/- standard deviation.
T = np.mean(G) - np.std(G)

_, B = cv2.threshold(G, T, 255, cv2.THRESH_BINARY_INV)
# Other methods of thresholding, not getting good results.
# _, B = cv2.threshold(G, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
# B = cv2.adaptiveThreshold(G, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 277, 25)

# Kernel used for morphology, cleaning white dots and join small empty black gaps
kernel = np.ones((3,3), np.uint8)
# Do morphology on image to fill in the gaps and decrease no. of outliers 
B = cv2.morphologyEx(B, cv2.MORPH_OPEN, kernel)


# Remove the last 25% of the image to avoid the white pixels down the bottom
# This is so required for second image because the intensity is the same for shark and water under sea.
h2 = int(B.shape[0] * 0.75)
# Set those pixels(75% - 100%) height to 0
B[h2:, :] = 0

plt.figure()
# Show the binary image after thresholding
plt.subplot(1,2,1)
plt.title('Binary Image')
plt.imshow(B, cmap='gray')

# cv2.imshow('Binary Image', B)

# Find the contours in the binary image, using RETR_EXTERNAL to find outer boundaries.
contours, h = cv2.findContours(B, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour of those contours found
largest_contour = findLargestContours(contours)

    
# Create a mask with only the largest contour (shark) with white background.
mask = img.copy()
mask[:, :] = (255, 255, 255)
mask = cv2.drawContours(mask, [largest_contour], 0, 0, -1)

# Show mask
plt.subplot(1,2,2)
plt.title('Shark mask')
mask_plt = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
plt.imshow(mask_plt)
# cv2.imshow('mask', mask)

# Add the mask with original image to crop the shark.
new_img = cv2.add(img, mask)


plt.figure()
# Cropped shark with white background
plt.subplot(1,2,1)
plt.title('Cropped with white bg')
newimg_plt = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
plt.imshow(newimg_plt)

# cv2.imshow('Cropped shark', new_img)

# Enhance the shark portion of the image
# Convert to LAB color space
new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(new_img)
# Use adaptive histogram equalization adjust L channel (luminance)
l = clahe.apply(l)
# Merge the channels back together
new_img = cv2.merge([l, a, b])
# Convert back to RGB for display.
new_img = cv2.cvtColor(new_img, cv2.COLOR_LAB2BGR)
# Show the image
plt.subplot(1,2,2)
plt.title('Enhanced Image')
newimg_plt = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
plt.imshow(newimg_plt)

# cv2.imshow('Enhanced shark', new_img)


# Option: automatically crop and rotate the image to contain only shark/shark
rect = cv2.minAreaRect(largest_contour)
center, size, angle = rect
center, size = tuple(map(int, center)), tuple(map(int, size))
rect_boxPoints = cv2.boxPoints(rect)
rect_boxPoints = np.array(rect_boxPoints, dtype=int)

# Get the four points of the rectangle
# The points goes from bottom right as p1
# bottom left p2, top left p3
# and top right p4
p1 = rect_boxPoints.take([0,1])
p2 = rect_boxPoints.take([2,3])
p3 = rect_boxPoints.take([4,5])
p4 = rect_boxPoints.take([6,7])

# Cropped shark/Shark using the coorindate of the points
cropped = new_img[p3[1]:p1[1], p2[0]:p4[0]]

plt.figure()
# Automatically cropped Shark.
plt.subplot(1,2,1)
plt.title('Cropped only shark')
cropped_plt = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
plt.imshow(cropped_plt)

# height and width of the cropped shark
c_h, c_w = cropped.shape[:2]
# Get the diagonal squared of the cropped image for rotation
d = int(math.sqrt(pow(c_w * 2, 2) + pow(c_h * 2, 2)))

# Transformation to center of new canvas
M = np.float32([[1, 0, d/2 - c_w/2], [0, 1, d/2 - c_h/2]])
T = cv2.warpAffine(cropped, M, (d, d), borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))


# Find the correct angle that the shark is horizontal
theta = abs(90 - abs(angle))

# Roatation from bottom of the cropped image to abs horizontal
M = cv2.getRotationMatrix2D(center=(d/2 + c_w/2, d/2 + c_h/2), angle=theta, scale=1)
rotated_result = cv2.warpAffine(T, M=M, dsize=(d, d), borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))

# Turn rotated image into gray for threshold new rect
gray_result = cv2.cvtColor(rotated_result, cv2.COLOR_BGR2GRAY)
# Find new threshold
_, B = cv2.threshold(gray_result, 240, 255, cv2.THRESH_BINARY_INV)

# Find the new contours from threshold
contours, h = cv2.findContours(B, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Get the largest contour
largest_contour = findLargestContours(contours)
# Get minAreaRect of the largest_contour (shark)
rect = cv2.minAreaRect(largest_contour)

# Get the co-ordinates of the rectangle
rect_boxPoints = np.array(cv2.boxPoints(rect), dtype=int)

# Get the four points of the rectangle
p1 = rect_boxPoints.take([0,1])
p2 = rect_boxPoints.take([2,3])
p3 = rect_boxPoints.take([4,5])
p4 = rect_boxPoints.take([6,7])

# Crop it with the four co-ordinates of the rectangle
final_result = rotated_result[p3[1]:p1[1], p2[0]:p4[0]]

plt.subplot(1,2,2)
plt.title('Final Result after rotation, cropped')
final_plt = cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB)
plt.imshow(final_plt)


# cv2.imshow('Final Result', final_result)
plt.show()

# Wait key pausing the program from finishing
cv2.waitKey(0)

