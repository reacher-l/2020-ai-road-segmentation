import cv2
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.ndimage.morphology import distance_transform_edt


def onehot_to_multiclass_edges(mask, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to an edgemap (K,H,W)

    """
    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

    channels = []
    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(1.0 - mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        dist = (dist > 0).astype(np.uint8)
        channels.append(dist)

    return np.array(channels)


def onehot_to_binary_edges(mask, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to a binary edgemap (H,W)

    """

    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

    edgemap = np.zeros(mask.shape[1:])
    for i in range(num_classes):
        # ti qu lun kuo
        dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(1.0 - mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edgemap += dist
    # edgemap = np.expand_dims(edgemap, axis=0)
    edgemap = (edgemap > 0).astype(np.uint8)*255
    return edgemap


def mask_to_onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    _mask = [mask == (i) for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)

if __name__ == '__main__':
    label = cv2.imread('/media/ws/新加卷1/wy/dataset/HUAWEI/data/labels/182/182_16_23.png',0)
    img = cv2.imread('/media/ws/新加卷1/wy/dataset/HUAWEI/data/images/182/182_16_23.png')
    oneHot_label = mask_to_onehot(label, 2)
    edge = onehot_to_binary_edges(oneHot_label, 2, 2) # #edge=255,background=0
    edge[:2, :] = 0
    edge[-2:, :] = 0
    edge[:, :2] = 0
    edge[:, -2:] = 0
    print(edge)
    print(np.unique(edge))
    print(edge.shape)
    cv2.imwrite('test.png',edge)
    cv2.namedWindow('1',0)
    cv2.namedWindow('2',0)
    cv2.namedWindow('3',0)
    cv2.imshow('1',label*255)
    cv2.imshow('2',edge)
    cv2.imshow('3',img)
    cv2.waitKey()