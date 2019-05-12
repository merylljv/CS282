import cv2
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
from scipy.misc import imread, imresize, imsave
import time

def find_vector_set(diff_image, new_size):
    i = 0
    j = 0
    vector_set = np.zeros((int(new_size[0] * new_size[1] / 25), 25))

    print('\nvector_set shape', vector_set.shape)

    while i < vector_set.shape[0]:
        while j < new_size[0]:
            k = 0
            while k < new_size[1]:
                block = diff_image[j:j + 5, k:k + 5]
                # print(i,j,k,block.shape)
                feature = block.ravel()
                vector_set[i, :] = feature
                k = k + 5
            j = j + 5
        i = i + 1

    mean_vec = np.mean(vector_set, axis=0)
    vector_set = vector_set - mean_vec

    return vector_set, mean_vec


def find_FVS(EVS, diff_image, mean_vec, new):
    i = 2
    feature_vector_set = []

    while i < new[0] - 2:
        j = 2
        while j < new[1] - 2:
            block = diff_image[i - 2:i + 3, j - 2:j + 3]
            feature = block.flatten()
            feature_vector_set.append(feature)
            j = j + 1
        i = i + 1

    FVS = np.dot(feature_vector_set, EVS)
    FVS = FVS - mean_vec
    print("\nFeature Vector Space size", FVS.shape)
    return FVS


def clustering(FVS, components, new):
    kmeans = KMeans(components, verbose=0)
    kmeans.fit(FVS)
    output = kmeans.predict(FVS)
    count = Counter(output)

    least_index = min(count, key=count.get)
    print(new[0], new[1])
    change_map = np.reshape(output, (new[0] - 4, new[1] - 4))

    return least_index, change_map


def find_PCAKmeans(imagepath1, imagepath2):
    print('\nOperating')

    image1 = cv2.imread(imagepath1, 0)
    image2 = cv2.imread(imagepath2, 0)

    image1 = cv2.adaptiveThreshold(image1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                   cv2.THRESH_BINARY, 11, 2)
    image2 = cv2.adaptiveThreshold(image2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                   cv2.THRESH_BINARY, 11, 2)

    new_size = np.asarray(image1.shape) / 5
    new_size = new_size.astype(int) * 5
    image1 = imresize(image1, (new_size)).astype(np.int16)
    image2 = imresize(image2, (new_size)).astype(np.int16)

    diff_image = abs(image1 - image2)
    #imsave('diff.jpg', diff_image)
    print('\nBoth images resized to ', new_size)

    vector_set, mean_vec = find_vector_set(diff_image, new_size)

    pca = PCA()
    pca.fit(vector_set)
    EVS = pca.components_

    FVS = find_FVS(EVS, diff_image, mean_vec, new_size)

    print('\nComputing K means')

    components = 3
    least_index, change_map = clustering(FVS, components, new_size)

    change_map[change_map == least_index] = 255
    change_map[change_map != 255] = 0

    change_map = change_map.astype(np.uint8)
    kernel = np.asarray(((0, 0, 1, 0, 0),
                         (0, 1, 1, 1, 0),
                         (1, 1, 1, 1, 1),
                         (0, 1, 1, 1, 0),
                         (0, 0, 1, 0, 0)), dtype=np.uint8)
    cleanChangeMap = cv2.erode(change_map, kernel)
    #plt.imshow(cv2.erode(cleanChangeMap, kernel), 'gray')
    #imsave("changemap.jpg", change_map)
    #imsave("cleanchangemap.jpg", cleanChangeMap)

    return diff_image, change_map, cleanChangeMap

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

def successrate(groundtruth, changemap):
    groundtruth_copy = groundtruth.copy()
    changemap_copy = changemap.copy()
    # Resize Ground Truth Image according to Change Map Image
    groundtruth_copy = cv2.resize(groundtruth_copy, (changemap_copy.shape[1], changemap_copy.shape[0]))
    # Force Ground Truth Image to have either 0 or 255 as pixel value
    groundtruth_copy_bin = cv2.threshold(groundtruth_copy, 1, 255, cv2.THRESH_BINARY)[1]
    # Convert Images to Float
    groundtruth_copy_bin = np.float32(groundtruth_copy_bin)
    changemap_copy = np.float32(changemap_copy)
    # Get Image Difference
    diff_image = (groundtruth_copy_bin - changemap_copy)
    tptn = np.count_nonzero(diff_image == 0)
    fp = np.count_nonzero(diff_image == -255)
    fn = np.count_nonzero(diff_image == 255)
    # Compute for Success Rate
    success_rate = tptn / (tptn + fp + fn)
    return round(success_rate,4)


# GENERATE OUTPUT

start = time.time()

scenes = 100
scene_idx = ["%.2d" % i for i in range(scenes)]
view_idx = ['0', '1', '2', '3', '4']
success_rates = []

for sc in scene_idx:
    for vw in view_idx:

        print('\nNow Processing Scene '+sc+' View '+vw)

        p1 = ('C:/Users/jlccjose/Desktop/MSCS/CS 282/Mini Project/AICDDataset/Images_NoShadow/Scene00'+sc+'_View0'+vw+'_moving.png')
        p2 = ('C:/Users/jlccjose/Desktop/MSCS/CS 282/Mini Project/AICDDataset/Images_NoShadow/Scene00'+sc+'_View0'+vw+'_target.png')
        p3 = ('C:/Users/jlccjose/Desktop/MSCS/CS 282/Mini Project/AICDDataset/GroundTruth/Scene00'+sc+'_View0'+vw+'_gtmask.png')

        diff, changemap, changemap_cl = find_PCAKmeans(p1, p2)
        changemap_cl_cb = clearborderobjects(changemap_cl,50)

        imsave(("output/diff_Scene00"+sc+"_View0"+vw+".jpg"), diff)
        imsave(("output/changemap_Scene00"+sc+"_View0"+vw+".jpg"), changemap)
        imsave(("output/changemap_cl_Scene00"+sc+"_View0"+vw+".jpg"), changemap_cl)
        imsave(("output/changemap_cl_cb_Scene00"+sc+"_View0"+vw+".jpg"), changemap_cl_cb)

        groundtruth = cv2.imread(p3, 0)
        success_rates = np.append(success_rates,[[sc,
                                                  vw,
                                                  successrate(groundtruth,changemap),
                                                  successrate(groundtruth,changemap_cl),
                                                  successrate(groundtruth,changemap_cl_cb)]])

end = time. time()
processing_time = (end-start)

print('\nProcessing Completed in ' + str(processing_time/3600) + ' Hours')

success_rates = success_rates.reshape((scenes*len(view_idx),3))
np.savetxt("success_rates.txt", success_rates,delimiter='\t',fmt='%1.4f')








