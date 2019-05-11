import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
from scipy.misc import imread, imresize, imsave

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
    return success_rate

# Success Rates below using Scene000_View00 (No Shadow)

successrate(groundtruth,changemap) # returns 0.9693349296819669
successrate(groundtruth,cleanchangemap) # returns 0.9912903443391454
successrate(groundtruth,cleanchangemap_clearborder) # returns 0.9982441570267444

