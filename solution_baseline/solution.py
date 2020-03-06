#!/usr/bin/env python3

import cv2
import numpy as np
from pathlib import Path
import histogram
from matplotlib import pyplot as plt
import os
import multiprocessing

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('clutter', help='directory containing clutter data')
parser.add_argument('out', help='directory to record output')
args = parser.parse_args()

from yaml import safe_load
with open('conf.yaml') as f:
    conf = safe_load(f.read())
objectsPath = Path(conf['objects'])

print('Loading images from {}'.format(str(objectsPath)))

def getName(team, obj):
    with open(str(team) + '/labels.yaml') as f:
        return safe_load(f.read())[str(obj).split('/')[-1]]

objects = {getName(team, obj): [cv2.imread(str(pic), cv2.IMREAD_UNCHANGED) for pic in obj.glob('**/*.png')] for team in objectsPath.iterdir() if team.is_dir() for obj in team.iterdir() if obj.is_dir()}

print('Found {} images of {} objects'.format(sum(len(objects[obj]) for obj in objects), len(objects)))

histograms = {}
for obj in objects:
    histograms[obj] = []
    for img in objects[obj]:
        alpha = img[:,:,3]
        img = img[:,:,:3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)
        data = histogram.getHistogramData(img, mask)
        #color = ('b','g','r')
        #for i,col in enumerate(color):
        #    plt.plot(data[i],color = col)
        #    plt.xlim([0,256])
        #plt.show()
        #cv2.waitKey()
        histograms[obj].append(data)

print('IoU for both first nerf: {}'.format(histogram.compareHistogramData(histograms['nerf'][0], histograms['nerf'][0])))
print('IoU for first and second nerf: {}'.format(histogram.compareHistogramData(histograms['nerf'][0], histograms['nerf'][1])))
print('IoU for nerf and catan: {}'.format(histogram.compareHistogramData(histograms['nerf'][0], histograms['catan'][0])))
print('IoU for nerf and pig: {}'.format(histogram.compareHistogramData(histograms['nerf'][0], histograms['pig'][0])))

# Get a mapping of object names to coco numbers
obj2id = {}
with open(args.clutter+'/names.txt') as f:
    mapping = [item for item in f.read().split('\n') if item]
for obj in mapping:
    obj2id[obj] = len(obj2id)
for obj in [obj for obj in objects if obj not in obj2id]:
    obj2id[obj] = len(obj2id)

def detectObject(args):
    xPos, yPos, height, width, clutter = args
    xOffset = min(span * xPos, width-windowDim)
    yOffset = min(span * yPos, height-windowDim)
    mask = np.zeros((height, width), np.uint8)
    mask[yOffset:yOffset+windowDim, xOffset:xOffset+windowDim] = np.ones((windowDim, windowDim), np.uint8)
    hist = histogram.getHistogramData(clutter, mask)
    scores = {obj: 0 for obj in objects}
    for obj in objects:
        for objHist in histograms[obj]:
            scores[obj] = max(scores[obj], histogram.compareHistogramData(hist, objHist))
    bestObj = max(scores, key=lambda obj: scores[obj])
    if scores[bestObj] > cutoff:
        print('Detected obj {}, confidence {}, at {},{}'.format(bestObj, scores[bestObj], xPos, yPos))
        return {'name': bestObj, 'score': scores[bestObj]}
    else:
        print('Did not find object, confidence {}, at {},{}'.format(scores[bestObj], xPos, yPos))
        return {'name': '', 'score': scores[bestObj]}

# Now we work through the input.
windowDim = 100
span = int(windowDim / 2)
cutoff = 0.3
p = multiprocessing.Pool(multiprocessing.cpu_count())
clutterPath = Path(args.clutter)
for clutterFile in clutterPath.glob('*.jpg'):
    print('Processing {}'.format(str(clutterFile)))
    clutter = cv2.imread(str(clutterFile))
    clutter = cv2.cvtColor(clutter, cv2.COLOR_BGR2HSV)
    height, width, _ = clutter.shape
    IDs = [[{'name': '', 'score': 0} for yPos in range(int(height/span))] for xPos in range(int(width/span))]
    # Dis get super ug real quick.
    for xPos in range(len(IDs)):
        IDs[xPos] = p.map(detectObject, [(xPos, yPos, height, width, clutter) for yPos in range(len(IDs[xPos]))])

    # Now that that's done, we can make sense of it all! Yippee!
    annotations = ''
    for xPos in range(len(IDs)):
        for yPos in range(len(IDs[xPos])):
            name = IDs[xPos][yPos]['name']
            if name:
                # Check if it's part of a region!
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
    
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    outfile = args.out+'/{}.txt'.format(os.path.splitext(os.path.basename(str(clutterFile)))[0])
    with open(outfile, 'w+') as f:
        f.write(annotations)
        print('Wrote output to {}'.format(outfile))
