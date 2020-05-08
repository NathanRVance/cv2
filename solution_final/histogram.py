#!/usr/bin/env python3

import cv2
import numpy as np
from skimage import feature
from sklearn import svm
from sklearn import ensemble
from sklearn import neighbors
import scipy

model = None
#modelOneClass = None
modelOneClasses = {}
histograms = None

# Augment: texture background, lighting, saturation

def registerGroundTruth(truth):
    global model
    #global modelOneClass
    global modelOneClasses
    global histograms
    global kernel
    #model = svm.LinearSVC(max_iter=10000)
    model = svm.NuSVC(kernel=kernel)
    #model = svm.NuSVC(kernel='sigmoid')
    #modelOneClass = svm.OneClassSVM(gamma='auto', kernel='sigmoid')
    #modelOneClass = svm.OneClassSVM(kernel='sigmoid')
    #modelOneClass = svm.OneClassSVM()
    #modelOneClass = svm.OneClassSVM(gamma='auto', kernel='poly')
    #modelOneClass = ensemble.IsolationForest()
    #modelOneClass = neighbors.LocalOutlierFactor(novelty=True)
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
    #modelOneClass.fit(data)

def getObject(image, mask=None):
    global model
    global modelOneClasses
    #global modelOneClass
    global histograms
    hist = getHistogramData(image, mask)
    #obj = model.predict(hist.reshape(1, -1))[0]
    #conf = max(compareHistogramData(hist, hist2) for hist2 in histograms[obj])
    #if modelOneClass.predict(hist.reshape(1, -1))[0] == -1:
    obj = model.predict(hist)[0]
    #conf = max(compareHistogramData(hist, hist2) for hist2 in histograms[obj])
    conf = 0
    for histogram in histograms[obj]:
        score = compareHistogramData(hist, histogram)
        if score > conf:
            conf = score
    
    score = modelOneClasses[obj].score_samples(hist)[0]
    #score = modelOneClass.score_samples(hist)[0]
    print('One class score: {}'.format(score))
    print('One class probability: {}'.format(np.exp(score)))
    #if modelOneClass.predict(hist)[0] == -1:
    #    conf /= 2
    conf *= np.exp(score)
    #cv2.imshow('Object', cv2.bitwise_and(image, image, mask=mask))
    #cv2.imshow('Match', img)
    #cv2.waitKey()
    return obj, conf

def getHistogramData(image, mask=None):
    getters = [getHistogramDataLBP, getHistogramDataColor, getVariance]
    #getters = [getHistogramDataLBP, getHistogramDataColor]
    data = np.array([])
    for getter in getters:
        data = np.append(data, getter(image, mask))
    return data.reshape(1, -1)

def getHistogramDataColor(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # We only care about channel 0 (h)
    image = image[:,:,0]
    hist = cv2.calcHist([image], [0], mask, [32], [0,256])
    #hist = cv2.calcHist([image], [0], mask, [256], [0,256])
    #hist = np.array([cv2.calcHist([image], [channel], mask, [256], [0,256]) for channel in range(image.shape[2])])
    hist = hist.reshape(1, -1)
    hist = hist.astype('float')
    hist /= (hist.sum() + 1e-1)
    return hist

def getHistogramDataLBP(image, mask=None):
    global lbpIndex
    # params = [ (numPoints, radius), ...]
    params = [(24, 8), (16, 4), (12, 2), (8, 1)]
    #params = [(24, 8), (16, 4), (8, 1)]
    params = [params[i] for i in range(len(params)) if i != lbpIndex]
    if mask is not None:
        image = cv2.bitwise_and(image, image, mask=mask)
    #print('About to cvtColor')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #print('About to calc lbp hist')
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
