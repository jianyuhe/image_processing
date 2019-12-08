# -*- coding: utf-8 -*-
"""
Created on  Nov 1 14:00:30 2019

Project Title: Text extraction.
Project Objective: Extract text in natural scene images
Team Members: Yuwei Chen, Minhui Chen, Jianyu He

"""
"""
Progrm description
	This code is using for detected the text in the natural environments
	This code using EAST, non-maximum suppression and Tesseract to implement
	EAST is text detector is a deep learning model, based on a novel architecture and training pattern.
	non-maximum suppression is using to remove the repetition text bounding box only keep one which is most likely text region
	Tesseract is using to convert text region to text and print it
	First input image, using EAST and non-maximum suppression to extract Text ROIS 
	And then using Tesseract OCR to convert ROI to extract the text
    Finally print text result on the screen
"""
from imutils.object_detection import non_max_suppression
from PIL import Image
import numpy as np
import cv2
import pytesseract
import easygui


    # two prameters scores: the possibility of text region 
    # geomtery: text bounding box position
def text_detector(scores, geometry):
    # The minimum probability of a detected text region
    min_confidence = 0.5
    
    (numRows,numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    
    for y in range(0, numRows):
        # extract the scores (probabilities)
        scoresData = scores[0,0,y]
        data0 = geometry[0,0,y]
        data1 = geometry[0,1,y]
        data2 = geometry[0,2,y]
        data3 = geometry[0,3,y]
        anglesData = geometry[0,4,y]
        
        for x in range(0, numCols):
            # if the scores doesn't have sufficient probability, ignore it
            if scoresData[x] < min_confidence:
                continue
                
            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            
            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            
            # use the geometry volume to derive the width and height
            # of the bounding box
            h = data0[x] + data2[x]
            w = data1[x] + data3[x]
            
            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * data1[x]) + (sin * data2[x]))
            endY = int(offsetY - (sin * data1[x]) + (cos * data2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
        
            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # return a tuple of the bounding boxes and associated confidences
    return (rects, confidences)


def text_recongnition():
    eastModel = "frozen_east_text_detection.pb"
    
    # set the new width and height and then determine the ratio in change for
    # both the width and height, both of them are multiples of 32
    newW = 320
    newH = 320
    
    #  The (optional) amount of padding to add to each ROI border
    # if find OCR result is incorrect you can try set padding to get bounding box bigger 
    # e.g 0.03 will incrase 3%
    padding = 0.03
    
    # in order to apply Tesseract v4 to OCR text we must supply
    # (1) a language, (2) an OEM flag of 4, indicating that the we
    # wish to use the LSTM neural net model for OCR, and finally
    # (3) an OEM value, in this case, 7 which implies that we are
    # treating the ROI as a single line of text
    config = ("-l eng --oem 1 --psm 7")  # chi_sim
    
    #read image
    f = easygui.fileopenbox()
    image = cv2.imread(f)
    #make a copy for image
    origI = image.copy()
    #get the image height and width
    h,w = image.shape[:2]
    
    # calculate ratios that will be used to scale bounding box coordinates
    ratioW = w/float(newW)
    ratioH = h/float(newH)
    

    # resize the image and grab the new image dimensions
    image = cv2.resize(image,(newW,newH))
    (IH,IW) = image.shape[:2]
    
    # define the two output layer names for the EAST detector model the first is the output probabilities
    # and the second can be used to derive the bounding box coordinates of text
    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    
    # load the pre-trained EAST text detector
    net = cv2.dnn.readNet(eastModel)
    
    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (IW, IH),(123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    # decode the predictions, then apply non-maxima suppression to
    # suppress weak, overlapping bounding boxes
    (rects, confidences) = text_detector(scores, geometry)
    # NMS effectively takes the most likely text regions, eliminating other overlapping regions
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # initialize the list of results to contain our OCR bounding boxes and text
    results = []

    # the bounding boxes represent where the text regions are, then recognize the text.
    # loop over the bounding boxes and process the results, preparing the stage for actual text recognition
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding boxes coordinates based on the respective ratios
        startX = int(startX * ratioW)
        startY = int(startY * ratioH)
        endX = int(endX * ratioW)
        endY = int(endY * ratioH)
        
        # in order to obtain a better OCR of the text we can potentially
        # add a bit of padding surrounding the bounding box -- here we
        # are computing the deltas in both the x and y directions
        dX = int((endX - startX) * padding)
        dY = int((endY - startY) * padding)

        # apply padding to each side of the bounding box, respectively
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(w, endX + (dX * 2))
        endY = min(h, endY + (dY * 2))
        
        # extract the actual padded ROI
        roi = origI[startY:endY, startX:endX]

        # use Tesseract v4 to recognize a text ROI in an image
        text = pytesseract.image_to_string(roi, config=config)
        
        # add the bounding box coordinates and actual text string to the results list
        results.append(((startX, startY, endX, endY), text))

    # sort the bounding boxes coordinates from top to bottom based on the y-coordinate of the bounding box
    results = sorted(results, key=lambda r:r[0][1])
    
    result = origI.copy()
    
    # loop over the results
    for ((startX, startY, endX, endY), text) in results:
        # display the text OCR'd by Tesseract
        print("{}\n".format(text))
        # draw the text and a bounding box surrounding the text region of the input image
        cv2.rectangle(result, (startX, startY), (endX, endY), (0, 0, 255), 2)

    # show the result image
    cv2.imshow("Text Recongnition", result)
    cv2.waitKey(0)

text_recongnition()