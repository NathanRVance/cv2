#!/usr/bin/env python3

import cv2
import numpy as np
from pathlib import Path
import histogram
import os
import multiprocessing

import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('train', help='directory containing training clutter data')
parser.add_argument('clutter', help='directory containing testing clutter data')
parser.add_argument('out', help='directory to record output')
parser.add_argument('--cutoff', '-c', help='confidence cutoff', default=0.15, type=float)
parser.add_argument('--lbp', help='lbp params index (0 to 3)', default=0, type=int)
parser.add_argument('--kernel', help='kernel to use', choices=('rbf', 'sigmoid', 'poly', 'linear'), default='rbf')
parser.add_argument('--parallel', help='use parallelism for inference', default=True, type=str2bool)
args = parser.parse_args()

# Initialize a few parameters
histogram.lbpIndex = args.lbp
histogram.kernel = args.kernel
windowDim = 100
span = int(windowDim / 2)
cutoff = args.cutoff

from yaml import safe_load
with open('conf.yaml') as f:
    conf = safe_load(f.read())
objectsPath = Path(conf['objects'])

print('Loading images from {}'.format(str(objectsPath)))

def getName(team, obj):
    with open(str(team) + '/labels.yaml') as f:
        return safe_load(f.read())[str(obj).split('/')[-1]]

objects = {getName(team, obj): [cv2.imread(str(pic), cv2.IMREAD_UNCHANGED) for pic in obj.glob('**/*.png')] for team in objectsPath.iterdir() if team.is_dir() for obj in team.iterdir() if obj.is_dir()}

# Get a mapping of object names to coco numbers
obj2id = {}
with open(args.train+'/names.txt') as f:
    mapping = [item for item in f.read().split('\n') if item]
for obj in mapping:
    obj2id[obj] = len(obj2id)
for obj in [obj for obj in objects if obj not in obj2id]:
    obj2id[obj] = len(obj2id)

print('Found {} images of {} objects'.format(sum(len(objects[obj]) for obj in objects), len(objects)))

def walk(img, mask=None):
    # utilize windowDim and span
    height, width, _ = img.shape
    if mask is None:
        mask = np.ones((height, width), np.uint8)
    maxDim = 300
    if max(height, width) > maxDim:
        img = cv2.resize(img, (width * maxDim // max(height, width), height * maxDim // max(height, width)))
        mask = cv2.resize(mask, (width * maxDim // max(height, width), height * maxDim // max(height, width)))
        height, width, _ = img.shape
    data = []
    if min(width, height) < windowDim:
        return [histogram.getHistogramData(img, mask)]
    for yPos in range(height//span):
        for xPos in range(width//span):
            xOffset = min(span * xPos, width-windowDim)
            yOffset = min(span * yPos, height-windowDim)
            mask2 = np.zeros((height, width), np.uint8)
            mask2[yOffset:yOffset+windowDim, xOffset:xOffset+windowDim] = np.ones((windowDim, windowDim), np.uint8)
            data.append(histogram.getHistogramData(img, cv2.bitwise_and(mask, mask2)))
    return data

    
def getForObj(img):
    alpha = img[:,:,3]
    img = img[:,:,:3]
    _, mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)
    #data = histogram.getHistogramData(img, mask)
    return walk(img, mask)

def getForClutter(clutter):
    print('Calculating truth histograms for {}'.format(clutter))
    hists = {}
    global obj2id
    img = cv2.imread(str(clutter))
    height, width, channels = img.shape
    with open(str(clutter).split('.')[0]+'.txt') as f:
        truth = [line.split(' ') for line in f.read().split('\n') if line]
    for line in truth:
        obj = next(obj for obj in obj2id if obj2id[obj] == int(line[0]))
        if obj not in hists:
            hists[obj] = []
        xc, yc, w, h = [float(num) for num in line[1:]] # xc, yc, w, h
        if w < 0.1 or h < 0.1:
            continue
        mask = np.zeros((height, width), np.uint8)
        #mask[int(height * (yc-h/2)):int(height * (yc+h/2)), int(width * (xc-w/2)), int(height * (yc+h/2))] = np.ones((int(height * h), int(width * w)), np.uint8)
        x = int(width * (xc-w/2))
        y = int(height * (yc-h/2))
        w = int(width * w)
        h = int(height * h)
        mask[y:y+h, x: x+w] = np.ones((h, w), np.uint8)
        #mask = cv2.rectangle(mask, (int(width * (xc-w/2)), int(height * (yc-h/2))), (int(width * (xc+w/2)), int(height * (yc+h/2))), (1), -1)
        #print('Displaying object {} at xc: {}, yc: {}, w: {}, h: {}:'.format(obj, xc, yc, w, h))
        #cv2.imshow('Masked', cv2.bitwise_and(img, img, mask=mask))
        #cv2.imshow('Original', img)
        #cv2.waitKey()
        #data = histogram.getHistogramData(img, mask)
        #hists[obj].append(data)
        hists[obj].extend(walk(img, mask))
    return hists

histograms = {}
with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
    for obj in objects:
        print('Calculating truth histograms for {}'.format(obj))
        histograms[obj] = []
        for data in p.map(getForObj, objects[obj]):
            histograms[obj].extend(data)
    for hists in p.map(getForClutter, Path(args.train).glob('*.jpg')):
        for obj in hists:
            histograms[obj].extend(hists[obj])

histogram.registerGroundTruth(histograms)

print('IoU for both first nerf: {}'.format(histogram.compareHistogramData(histograms['nerf'][0], histograms['nerf'][0])))
print('IoU for first and second nerf: {}'.format(histogram.compareHistogramData(histograms['nerf'][0], histograms['nerf'][1])))
print('IoU for nerf and catan: {}'.format(histogram.compareHistogramData(histograms['nerf'][0], histograms['catan'][0])))
print('IoU for nerf and pig: {}'.format(histogram.compareHistogramData(histograms['nerf'][0], histograms['pig'][0])))
ob, conf = histogram.getObject(objects['nerf'][0])
print('Predicted object for nerf: {}, confidence: {}'.format(ob, conf))
ob, conf = histogram.getObject(objects['brush'][0])
print('Predicted object for brush: {}, confidence: {}'.format(ob, conf))
ob, conf = histogram.getObject(objects['glasses'][0])
print('Predicted object for glasses: {}, confidence: {}'.format(ob, conf))
#exit()

def processFile(clutterFile):
    print('Processing {}'.format(str(clutterFile)))
    clutter = cv2.imread(str(clutterFile))
    height, width, _ = clutter.shape
    if max(height, width) > 1080:
        clutter = cv2.resize(clutter, (width * 1080 // max(height, width), height * 1080 // max(height, width)))
    height, width, _ = clutter.shape
    IDs = [[{'name': '', 'score': 0} for yPos in range(int(height/span))] for xPos in range(int(width/span))]
    for xPos in range(len(IDs)):
        for yPos in range(len(IDs[xPos])):
            xOffset = min(span * xPos, width-windowDim)
            yOffset = min(span * yPos, height-windowDim)
            mask = np.zeros((height, width), np.uint8)
            mask[yOffset:yOffset+windowDim, xOffset:xOffset+windowDim] = np.ones((windowDim, windowDim), np.uint8)
            bestObj, score = histogram.getObject(clutter, mask)
            if score > cutoff:
                print('Detected obj {}, confidence {}, at {},{}'.format(bestObj, score, xPos, yPos))
                IDs[xPos][yPos] = {'name': bestObj, 'score': score}
            else:
                print('Did not find object, confidence {}, at {},{}'.format(score, xPos, yPos))
                IDs[xPos][yPos] = {'name': '', 'score': score}

    # Now that that's done, we can make sense of it all! Yippee!
    print('Processing results')
    annotations = ''
    for xPos in range(len(IDs)):
        for yPos in range(len(IDs[xPos])):
            name = IDs[xPos][yPos]['name']
            if name:
                # Check if it's part of a region
                def walk(x, y, name):
                    if x < 0 or y < 0 or x >= len(IDs) or y >= len(IDs[x]) or IDs[x][y]['name'] != name:
                        return []
                    parts = [(x*span, y*span)]
                    IDs[x][y]['name'] = ''
                    parts.extend(walk(x-1, y, name))
                    parts.extend(walk(x+1, y, name))
                    parts.extend(walk(x, y-1, name))
                    parts.extend(walk(x, y+1, name))
                    parts.extend(walk(x+1, y+1, name))
                    parts.extend(walk(x+1, y-1, name))
                    parts.extend(walk(x-1, y+1, name))
                    parts.extend(walk(x-1, y-1, name))
                    return parts
                region = walk(xPos, yPos, name)
                px = [x for x,_ in region]
                py = [y for _,y in region]
                minX = min(px) / width
                minY = min(py) / height
                maxX = max(px) / width + span/width
                maxY = max(py) / height + span/height
                xc = (minX + maxX) / 2
                yc = (minY + maxY) / 2
                w = maxX - minX
                h = maxY - minY
                assert w <= width and h <= height and xc < width and yc < height
                annotations += '{} {} {} {} {}\n'.format(obj2id[name], xc, yc, w, h)
    print('Writing to file')
    outfile = args.out+'/{}.txt'.format(os.path.splitext(os.path.basename(str(clutterFile)))[0])
    with open(outfile, 'w+') as f:
        f.write(annotations)
        print('Wrote output to {}'.format(outfile))

# Now we work through the input.
if not os.path.exists(args.out):
    os.makedirs(args.out)
clutterPath = Path(args.clutter)
if args.parallel:
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        p.map(processFile, clutterPath.glob('*.jpg'))
else:
    for fname in clutterPath.glob('*.jpg'):
        processFile(fname)
