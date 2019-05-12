from collections import Counter
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

import ClearBorderObjects as clear_border
import SuccessRate as sr


def find_vector_set(diff_image, new_size):
   
    i = 0
    j = 0
    vector_set = np.zeros((int(new_size[0] * new_size[1] / 25), 25))
    
    while i < vector_set.shape[0]:
        while j < new_size[0]:
            k = 0
            while k < new_size[1]:
                block   = diff_image[j:j+5, k:k+5]
                #print(i,j,k,block.shape)
                feature = block.ravel()
                vector_set[i, :] = feature
                k = k + 5
            j = j + 5
        i = i + 1
        
            
    mean_vec   = np.mean(vector_set, axis = 0)    
    vector_set = vector_set - mean_vec
    
    return vector_set, mean_vec
    
  
def find_FVS(EVS, diff_image, mean_vec, new):
    
    i = 2 
    feature_vector_set = []
    
    while i < new[0] - 2:
        j = 2
        while j < new[1] - 2:
            block = diff_image[i-2:i+3, j-2:j+3]
            feature = block.flatten()
            feature_vector_set.append(feature)
            j = j+1
        i = i+1
        
    FVS = np.dot(feature_vector_set, EVS)
    FVS = FVS - mean_vec

    return FVS

def clustering(FVS, components, new):
    
    kmeans = KMeans(components, verbose = 0)
    kmeans.fit(FVS)
    output = kmeans.predict(FVS)
    count  = Counter(output)

    least_index = min(count, key = count.get)            
    change_map  = np.reshape(output,(new[0] - 4, new[1] - 4))
    
    return least_index, change_map

   
def find_PCAKmeans(path, file_image1, file_image2):
    
    print('Operating:', file_image1.replace('_moving.png', ''))
    
    image1 = cv2.imread(path + file_image1, 0)
    image2 = cv2.imread(path + file_image2, 0)
    
    image1 = cv2.adaptiveThreshold(image1,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
    image2 = cv2.adaptiveThreshold(image2,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)

    image1 = cv2.bilateralFilter(image1,9,75,75)
    image2 = cv2.bilateralFilter(image2,9,75,75)
    
    new_size = np.asarray(image1.shape) / 5
    new_size = new_size.astype(int) * 5
    image1 = cv2.resize(image1, (new_size[1], new_size[0])).astype(np.int16)
    image2 = cv2.resize(image2, (new_size[1], new_size[0])).astype(np.int16)
    diff_image = abs(image1 - image2)   

    fig = plt.figure(figsize=(image1.shape[0] / 10, int(image1.shape[1]) / 50))
    ax1 = fig.add_subplot(151)
    ax1.imshow(diff_image, 'gray')

    kernel     = np.asarray(((0,1,0),
                             (1,1,1),
                             (0,1,0)), dtype=np.uint8)
    diff_image = cv2.morphologyEx(diff_image, cv2.MORPH_CLOSE, kernel)
    ax2 = fig.add_subplot(152)
    ax2.imshow(diff_image, 'gray')

    vector_set, mean_vec = find_vector_set(diff_image, new_size)
    
    pca     = PCA()
    pca.fit(vector_set)
    EVS = pca.components_
        
    FVS     = find_FVS(EVS, diff_image, mean_vec, new_size)
    
    # computing k means
    components = 3
    least_index, change_map = clustering(FVS, components, new_size)
    
    change_map[change_map == least_index] = 255
    change_map[change_map != 255] = 0
    
    change_map = change_map.astype(np.uint8)
    kernel     = np.asarray(((0,0,1,0,0),
                             (0,1,1,1,0),
                             (1,1,1,1,1),
                             (0,1,1,1,0),
                             (0,0,1,0,0)), dtype=np.uint8)
    clean_change_map = cv2.erode(change_map,kernel)
    cleanchangemap_clearborder = clear_border.clearborderobjects(clean_change_map,10)

    ax3 = fig.add_subplot(153)
    ax3.imshow(change_map, 'gray')
    ax4 = fig.add_subplot(154)
    ax4.imshow(clean_change_map, 'gray')
    ax5 = fig.add_subplot(155)
    ax5.imshow(cleanchangemap_clearborder, 'gray')
    plt.savefig('changemap/' + file_image1.replace('_moving', ''), transparent=True)
    plt.close()
    groundtruth = cv2.imread('data/GroundTruth/' + file_image1.replace('moving', 'gtmask'), 0)
    changemap_dict = {'diff': diff_image, 'changemap': change_map,
                      'clean': clean_change_map, 'noborder': cleanchangemap_clearborder}
    sr.successrate(file_image1.replace('_moving.png', ''), groundtruth, changemap_dict)


if __name__ == "__main__":
    start = datetime.now()
    path = os.getcwd()
    try:  
        os.mkdir(path + '/changemap/')
    except OSError:  
        pass

    scene_idx = ["%.4d" % i for i in range(100)]
    view_idx = ['00', '01', '02', '03', '04']
    for scene in scene_idx:
        for view in view_idx:
            img1 = 'Scene' + scene + '_View' + view + '_moving.png'
            img2 = 'Scene' + scene + '_View' + view + '_target.png'
            find_PCAKmeans('data/Images_NoShadow/', img1, img2)
    print('runtime =', str(datetime.now() - start))