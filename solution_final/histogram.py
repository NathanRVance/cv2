#!/usr/bin/env python3

import cv2
import numpy as np
from skimage import feature
from sklearn import svm
from sklearn import ensemble
from sklearn import neighbors
import scipy

model = None
modelOneClasses = {}
histograms = None

def registerGroundTruth(truth):
    global model
    global modelOneClasses
    global histograms
    global kernel
    model = svm.NuSVC(kernel=kernel)
    for obj in truth:
        modelOneClasses[obj] = neighbors.LocalOutlierFactor(novelty=True)
    histograms = truth
    data = []
    labels = []
    for obj in truth:
        objData = []
        for hist in truth[obj]:
            data.append(hist[0])
            objData.append(hist[0])
            labels.append(obj)
        modelOneClasses[obj].fit(objData)
    print('Fitting model to data')
    model.fit(data, labels)

def getObject(image, mask=None):
    global model
    global modelOneClasses
    global histograms
    hist = getHistogramData(image, mask)
    obj = model.predict(hist)[0]
    conf = 0
    for histogram in histograms[obj]:
        score = compareHistogramData(hist, histogram)
        if score > conf:
            conf = score
    
    score = modelOneClasses[obj].score_samples(hist)[0]
    conf *= np.exp(score)
    return obj, conf

def getHistogramData(image, mask=None):
    getters = [getHistogramDataLBP, getHistogramDataColor, getVariance]
    data = np.array([])
    for getter in getters:
        data = np.append(data, getter(image, mask))
    return data.reshape(1, -1)

def getHistogramDataColor(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # We only care about channel 0 (h)
    image = image[:,:,0]
    hist = cv2.calcHist([image], [0], mask, [32], [0,256])
    hist = hist.reshape(1, -1)
    hist = hist.astype('float')
    hist /= (hist.sum() + 1e-1)
    return hist

def getHistogramDataLBP(image, mask=None):
    global lbpIndex
    # params = [ (numPoints, radius), ...]
    params = [(24, 8), (16, 4), (12, 2), (8, 1)]
    params = [params[i] for i in range(len(params)) if i != lbpIndex]
    if mask is not None:
        image = cv2.bitwise_and(image, image, mask=mask)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hists = []
    for numPoints, radius in params:
        lbp = feature.local_binary_pattern(image, numPoints, radius, method='uniform')
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, numPoints+3), range=(0, numPoints+2))
        hist = hist.astype('float')
        hist /= (hist.sum() + 1e-7)
        hists.extend(hist)
    hists = np.array(hists).reshape(1, -1)
    return hists

def getVariance(image, mask=None):
    masked = cv2.bitwise_and(image, image, mask=mask)
    return np.array(min(1.0, 1/(scipy.ndimage.measurements.variance(masked)+0.00001))).reshape(1, -1)

def compareHistogramData(data1, data2):
    histIntersect = sum(min(datum1, datum2) for datum1, datum2 in zip(data1[0], data2[0]))
    histUnion = sum(max(datum1, datum2) for datum1, datum2 in zip(data1[0], data2[0]))
    return histIntersect / histUnion
