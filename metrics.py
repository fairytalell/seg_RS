#!conda env python
# -*- encoding: utf-8 -*-
import numpy as np

def compute_global_accuracy(pred, label):
    '''
    Compute the average segmentation accuracy across all classes,
    Input [HW] or [HWC] label
    '''
    count_mat = pred == label
    return np.sum(count_mat) / np.prod(count_mat.shape)
