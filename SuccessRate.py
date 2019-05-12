import cv2
import numpy as np
import pandas as pd

def successrate(file_name, groundtruth, changemap_dict):    
    rate = pd.DataFrame({'scene_view': [file_name]})

    for changemap_key in changemap_dict.keys():
        changemap = changemap_dict.get(changemap_key)
        # Resize Ground Truth Image according to Change Map Image
        groundtruth = cv2.resize(groundtruth, (changemap.shape[1], changemap.shape[0]))
        # Force Ground Truth Image to have either 0 or 255 as pixel value
        groundtruth_bin = cv2.threshold(groundtruth, 1, 255, cv2.THRESH_BINARY)[1]
        # Convert Images to Float
        groundtruth_bin = np.float32(groundtruth_bin)
        changemap = np.float32(changemap)
        # Get Image Difference
        diff_image = (groundtruth_bin - changemap)
        tptn = np.count_nonzero(diff_image == 0)
        tp = np.count_nonzero(groundtruth[diff_image == 0])
        rate[changemap_key + '_tp'] = tp
        tn = tptn - tp
        rate[changemap_key + '_tn'] = tn
        fp = np.count_nonzero(diff_image == -255)
        rate[changemap_key + '_fp'] = fp
        fn = np.count_nonzero(diff_image == 255)
        rate[changemap_key + '_fn'] = fn
        if tp+fp != 0:
            precision = tp / (tp+fp)
        else:
            precision = 0
        rate[changemap_key + '_precision'] = precision
        if tp+fn != 0:
            recall = tp / (tp+fn)
        else:
            recall = 0
        rate[changemap_key + '_recall'] = recall
        acc = (tp+tn) / (tp+tn+fp+fn)
        rate[changemap_key + '_accuracy'] = acc
        
    try:
        metrics = pd.read_csv('metrics.csv')
    except:
        metrics = pd.DataFrame()
    
    metrics = metrics.append(rate)
    metrics = metrics.drop_duplicates('scene_view')
    metrics.to_csv('metrics.csv', index=False)