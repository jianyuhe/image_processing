import cv2
import numpy as np 
from matplotlib import pyplot as plt
from random import randint
import easygui


'''
    Program Name: SPOT THE BALL -assignment2
    Studnet Name: Jianyu He
    Student Number: D17124591
    Description
     Using opencv techniques to sucessfully crop the ball and Replace the hole suitable color
        project idea:
        I compare 3 image to find what is the similar of balls, 1. ball is biggest white in the image,
        2. ball is circle
        I have 2 ideas for the project
        first idea is I can use inrange function to find white color mask image, then use HoughCircles
        function to find all circle of mask image, after find biggest circle which is location of ball,
        then i can draw the biggest circle on the new black mask image, final I can use inpaint function,
        original image and final mask image to replace the hole which is location of ball.
        
        second idea is I can use houghcicles function to find all circle of original image, then I can use
        x,y and radius to find mini rect of circle to contain circle, after I use img[] to got each rect,
        then use inrange function to create white color mask image for each img[], final I can use loop to
        detect each color of pixel, if the color is white that means img[] contain white circle, as result I
        can find biggest area of rect it contain ball, I can through rect to calculate x, y and radius of ball,
        draw the ball on new black mask image, use inpaint function to fix hole.
        
        I choose first idea for my project.
    
        Implementation:
        First, change color space of original image from bgr to rgb, becasue I will shows image on plt.
        the image is converted to HSV colorspace to prepare for increasing contrast/intensity.
        use inrange function to output mask image which is area of white color, after use erorsion and dilate
        to clean the small noise and fix circle.
        
        then use HoughCircles to find specified circle from mask image, I can know x,y and radius of circle.
        after create a new balck mask image, then draw the specified circle in the black mask image, output the
        final mask image.
        
        final, use inpaint function, origin image and final mask image to replace the circle with suitable color
        
        I use plt to show all of the image
        total 6 image in plt it will be 2*3 form
        
        the results of golf image is not contain all ball so show the final image is not very good, if I want better
        solution I can increase radius of circle.


    Date: 17/11/19
'''




# return suitable mask image can use to find white circle
def suit_mask(img):
    # Convert the image into hsv colorspace for contrast
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
#    declare red, green, blue value of lower white color 
    lower_white = np.array([0,0,168])
    
    #    declare red, green, blue value of upper white color
    upper_white = np.array([172,111,255])
    
# use hsv image, and inrange function to output mask image can use to find area of white color
    wmask = cv2.inRange(hsv, lower_white, upper_white)
    
#     shows origin white color mask image in plt
    plt.subplot(2,3,2)
    plt.title('white color mask image')
    plt.imshow(wmask,cmap='gray')

    # Kernel used for erode and dilate, cleaning white dots and join small empty black gaps
    kernel = np.ones((5,5), np.uint8)
    
#     use original mask image and erode function to create mask image which is clean the small noise 3 times
    erorsion_img = cv2.erode(wmask, kernel, iterations=3)

# show mask image after erorsion in plt
    plt.subplot(2,3,3)
    plt.title('erorsion image')
    plt.imshow(erorsion_img,cmap='gray')
    
# use erorsion mask image and dilate function to create mask image which is fix the circle 5 times
    dilate_img = cv2.dilate(erorsion_img, kernel, iterations=5)
    
    # show mask image after dilate in plt
    plt.subplot(2,3,4)
    plt.title('dilate image')
    plt.imshow(dilate_img,cmap='gray')
    
#     return the final mask image
    return dilate_img




# Load image from project folder
image_file = easygui.fileopenbox()

# Read image from file
img = cv2.imread(image_file)

# change color space from bgr to rgb can use for plt
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# read image from file change it to gray
img1 = cv2.imread(image_file,0)

# shows origin image on plt
plt.subplot(2,3,1)
plt.title('origin image')
plt.imshow(img)

# create suit mask for original image
simg = suit_mask(img)


# use houghcircles function to detect specified circle from mask
# simg is mask image to find area of white color from origin image
# cv2.HOUGH_GRADIENT is Defines the method to detect circles in images.
# 1 is the inverse ratio of the accumulator resolution to the image resolution
# 100 is Minimum distance between the center (x, y) coordinates of detected circles
# np.array([]),100 is Canny edge detection requires two parameters — minVal and maxVal.
# 5 is the accumulator threshold for the candidate detected circles. if this value is 100 can only detect 100% circle
# 20 is Minimum circle radius.
# 50 is Maximum circle radius.
# output circle will contain x,y and radius of all circle, it can use to find all circle from mask image
circles = cv2.HoughCircles(simg, cv2.HOUGH_GRADIENT, 1, 100, np.array([]),100, 5, 20, 50)

# below is houghcircles function only use to find specified circle of football image
# circles = cv2.HoughCircles(img1, cv2.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 1, 20)

# below is houghcircles function only use to find specified circle of snooker image
# circles = cv2.HoughCircles(img1, cv2.HOUGH_GRADIENT, 1, 100, np.array([]), 100, 43, 1, 40)

# below is houghcircles function only use to find specified circle of golf image
# circles = cv2.HoughCircles(img1, cv2.HOUGH_GRADIENT, 1, 100, np.array([]), 100, 43, 33, 50)





# use gray origin image to create a mask image which is all black
ret,mask = cv2.threshold(img1,0,0,cv2.THRESH_BINARY_INV)



# so only draw specified circle in black mask image
# declare x,y and radius for biggest circle 
x = 0
y = 0
radius = 0
# use loop find x, y and radius of biggest circle
for i in circles[0,:]:
    if i[2]>radius:
        x = i[0]
        y = i[1]
        radius = i[2]

# draw biggest circle in black mask image.
cv2.circle(mask, (x,y), radius, (255,255,255), -1, cv2.LINE_AA)

# show final mask image which is only drow a specified circle
plt.subplot(2,3,5)
plt.title('final mask image for specified circle')
plt.imshow(mask, cmap='gray')
 
# below is different way to fill specified circle, it use color of random pixel which is around circle to fill circle,
# but the result is not very good
# h,w,c = newi.shape
# for row in range(h):
#     for col in range(w):
#         if all(newi[row,col] == [0,0,0]):
#             newpix = newi[int(circles[0][0][0]+circles[0][0][2] + randint(1,20)), int(circles[0][0][1])]
#             newi[row,col] = newpix

# use inpaint function, original image and final mask image to fill suitable color for circle then output final image
# which is no ball
dst = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)

# shows final image without ball in plt
plt.subplot(2,3,6)
plt.title('final image without ball')
plt.imshow(dst)

plt.show()
cv2.waitKey(0)


