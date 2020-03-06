#!/usr/bin/env python3

import cv2
import numpy as np

def getHistogramData(image, mask=None):
    return [cv2.calcHist([image], [channel], mask, [256], [0,256]) for channel in range(image.shape[2])]

# Basically, an IoU for histograms.
def compareHistogramData(data1, data2):
    # First, make them the same "size"
    # Each channel should account for same number of items
    mag1 = sum(data1[0])
    mag2 = sum(data2[0])
    if mag2 == 0:
        return 0
    # Adjust data2 to match data1
    data2 = [[datum * mag1/mag2 for datum in channel] for channel in data2]
    histIntersect = [[min(datum1, datum2) for datum1, datum2 in zip(channel1, channel2)] for channel1, channel2 in zip(data1, data2)]
    histUnion = [[max(datum1, datum2) for datum1, datum2 in zip(channel1, channel2)] for channel1, channel2 in zip(data1, data2)]
    return sum(sum(channel) for channel in histIntersect) / sum(sum(channel) for channel in histUnion)
