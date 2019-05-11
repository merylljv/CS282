import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
from scipy.misc import imread, imresize, imsave

def clearborderobjects(imgBW, radius):
    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    bin, contours, hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Get dimensions of image
    imgRows = imgBW.shape[0]
    imgCols = imgBW.shape[1]
    # Initialize list of contours that are within the radius of the border
    contourList = []
    # For each contour, check if any point of the contour is within the radius of the border
    for idx in np.arange(len(contours)):
        cnt = contours[idx]
        for pt in cnt:
            rowCnt = pt[0][1]
            colCnt = pt[0][0]
            # If any point is within the radius of the border, add the contour to contourList
            check1 = (rowCnt >= 0 and rowCnt < radius) or (rowCnt >= imgRows-1-radius and rowCnt < imgRows)
            check2 = (colCnt >= 0 and colCnt < radius) or (colCnt >= imgCols-1-radius and colCnt < imgCols)
            if check1 or check2:
                contourList.append(idx)
                break
    # Remove each contour in contourList
    for idx in contourList:
        cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)
    return imgBWcopy

cleanchangemap_clearborder = clearborderobjects(cleanchangemap,10)
cv2.imshow('Clean Change Map (Without Border Objects)',cleanchangemap_clearborder)
