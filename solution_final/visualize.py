#!/usr/bin/env python3

import cv2
import numpy as np
import re

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('detected', help='directory with detected bounding boxes')
parser.add_argument('images', help='directory containing jpg images')
args = parser.parse_args()

with open(args.images+'/names.txt') as f:
    names = [name for name in f.read().split('\n') if name]

from pathlib import Path
detected = Path(args.detected)

for fname in detected.glob('*.txt'):
    if not re.findall(r'\d+\.', str(fname)):
        continue
    scene = cv2.imread('{}/{}.jpg'.format(args.images, str(fname).split('/')[-1].split('.')[0]))
    height, width, _ = scene.shape
    if max(height, width) > 1080:
        scene = cv2.resize(scene, (width * 1080 // max(height, width), height * 1080 // max(height, width)))
    height, width, _ = scene.shape
    with open(fname) as f:
        boxes = [line for line in f.read().split('\n') if line]
    for box in boxes:
        objID, x, y, w, h = box.split(' ')
        objID = int(objID)
        x = int(float(x) * width)
        y = int(float(y) * height)
        w = int(float(w) * width)
        h = int(float(h) * height)
        cv2.rectangle(scene, (x-w//2, y-h//2), (x+w//2, y+h//2), (0, 255, 0), 2)
        cv2.putText(scene, names[objID], (x, y+h//2+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
    print('Displaying for file {}'.format(fname))
    cv2.imshow('Results', scene)
    cv2.waitKey()

