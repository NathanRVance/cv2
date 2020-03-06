#!/usr/bin/env python3

import cv2
import numpy as np
import os
from pathlib import Path
import random
import imutils
import json
import base64

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--save', help='save images to path, e.g., data/out.jpg')
parser.add_argument('-n', '--number', help='number of images to generate. If used with --save, image ID is appended to filename, e.g. data/out1.jpg', type=int, default=1)
parser.add_argument('-o', '--objects', help='maximum number of objects selected for each image', type=int, default=7)
args = parser.parse_args()

from yaml import safe_load
with open('conf.yaml') as f:
    conf = safe_load(f.read())
objectsPath = Path(conf['objects'])

print('Loading images from {}'.format(str(objectsPath)))

def getName(team, obj):
    with open(str(team) + '/labels.yaml') as f:
        return safe_load(f.read())[str(obj).split('/')[-1]]

objects = {str(team): {getName(team, obj): [cv2.imread(str(pic), cv2.IMREAD_UNCHANGED) for pic in obj.glob('**/*.png')] for obj in team.iterdir() if obj.is_dir()} for team in objectsPath.iterdir() if team.is_dir()}

print('Found {} images of {} objects from {} teams'.format(sum(sum(len(objects[team][obj]) for obj in objects[team]) for team in objects), sum(len(objects[team]) for team in objects), len(objects)))

numObjects = min(args.objects, sum(len(objects[team]) for team in objects))

for sceneID in range(args.number):
    selected = {obj: random.choice(objects[team][obj]) for (team, obj) in random.sample([(team, obj) for team in objects for obj in objects[team]], numObjects)}
    
    height = 720
    width = 1080
    scene = np.zeros((height,width,3), np.float)
    scene[:] = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
    
    level = numObjects
    scaleFactor = .95
    border=0
    labels={"version": "1.3.3.4", "flags": {}, "shapes": [], "imageHeight": height, "imageWidth": width}
    pastMasks=[]
    for name, obj in selected.items():
        scale = scaleFactor ** level
        level -= 1
        obj = imutils.rotate_bound(obj, random.randint(0, 360))
        h, w, _ = obj.shape
        if max(h, w) > min(height, width) - border*2:
            scale *= (min(height, width)-border*2) / max(h, w)
        obj = cv2.resize(obj, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w, _ = obj.shape
        destX = random.randint(border, width - w - border)
        destY = random.randint(border, height - h - border)
        print('Adding {} at {},{}'.format(name, destX, destY))
        alpha = obj[:,:,3]
        obj = obj[:,:,:3]
        _, mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY) # Was obj not alpha
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        pastMasks.append({'mask': mask, 'offset': (destX, destY), 'name': name})
        megamask = np.zeros(obj.shape, np.float)
        for channel in range(obj.shape[2]):
            megamask[:,:,channel] = mask
        mask = megamask
        #maskGray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        #contours, hierarchy = cv2.findContours(maskGray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #print('contours: {}, hierarchy: {}'.format(contours, hierarchy))
        mask = mask.astype(float)
        obj = obj.astype(float)
        obj = cv2.multiply(mask/255, obj)
        scene[destY:destY+h,destX:destX+w] = cv2.multiply(1.0-mask/255, scene[destY:destY+h,destX:destX+w])
        scene[destY:destY+h,destX:destX+w] = cv2.add(scene[destY:destY+h,destX:destX+w], obj)
        # Also save labels!
        #label = {"label": name, "group_id": None, "shape_type": "polygon", "flags": {}, "points": [[int(point[0][0]+destX), int(point[0][1]+destY)] for point in contours[0]]}
        #labels["shapes"].append(label)

    for i, current in enumerate(pastMasks):
        currMask = cv2.bitwise_not(current['mask'])
        #currMask = current['mask']
        foreScene = np.ones((height,width), np.uint8)
        currH, currW = currMask.shape
        currX, currY = current['offset']
        foreScene[currY:currY+currH,currX:currX+currW] = currMask

        for behind in pastMasks[:i]:
            behMask = behind['mask']
            behX, behY = behind['offset']
            # Move currMask based on offset, and then crop
            tmpScene = np.zeros((height,width), np.uint8)
            behH, behW = behMask.shape
            tmpScene[behY:behY+behH,behX:behX+behW] = behMask
            tmpScene = cv2.bitwise_and(foreScene, tmpScene)
            behind['mask'] = tmpScene[behY:behY+behH,behX:behX+behW]

    for maskData in pastMasks:
        #maskGray = cv2.cvtColor(maskData['mask'], cv2.COLOR_BGR2GRAY)
        #maskGray = cv2.cvtColor(maskData['mask'][:,:,0], cv2.COLOR_BGR2GRAY)
        #maskGray = cv2.cvtColor(maskData['mask'].astype(np.uint8), cv2.COLOR_BGR2GRAY)
        #contours, hierarchy = cv2.findContours(maskGray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(maskData['mask'], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        destX, destY = maskData['offset']
        for c, h in zip(contours, hierarchy[0]):
            #print(c)
            #print('Hierarchy for {}: {}'.format(maskData['name'], h))
            if h[3] < 0:
                label = {"label": maskData['name'], "group_id": None, "shape_type": "polygon", "flags": {}, "points": [[int(point[0][0]+destX), int(point[0][1]+destY)] for point in c]}
                labels["shapes"].append(label)
            else:
                print('Ignoring {} because index 3 is {}'.format(h, h[3]))
    
    if args.save:
        savePath = args.save
        if args.number > 1:
            path, ext = savePath.split('.')
            savePath = '{}{}.{}'.format(path, sceneID, ext)
            directory = os.path.dirname(path)
            if not os.path.exists(directory):
                os.makedirs(directory)
        print('Saving scene to {}'.format(savePath))
        cv2.imwrite(savePath, scene)
        labels["imagePath"] = savePath.split('/')[-1]
        _, scene = cv2.imencode('.'+savePath.split('.')[-1], scene)
        labels["imageData"] = base64.b64encode(scene).decode()
        with open(savePath.split('.')[0]+'.json', 'w') as f:
            f.write(json.dumps(labels))
    else:
        cv2.imshow('Clutter', scene/255)
        cv2.waitKey()
