#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('detected', help='directory with detected bounding boxes')
parser.add_argument('truth', help='directory with ground truth')
args = parser.parse_args()

from pathlib import Path
truthPath = Path(args.truth)
detectedPath = Path(args.detected)

with open(args.truth+'/names.txt') as f:
    mappings = [item for item in f.read().split('\n') if item]
categories = [{'supercategory': 'none', 'name': name, 'id': mappings.index(name)} for name in mappings]

import re

DIM = 1000

def dataset2json(path):
    dataset = {'type': 'instances', 'images': [], 'categories': categories, 'annotations': []}
    for truthFile in path.glob('*.txt'):
        if not re.findall(r'\d+\.', str(truthFile)):
            continue
        fileID = int(re.findall(r'\d+\.', str(truthFile))[0][:-1])
        dataset['images'].append({'file_name': str(truthFile), 'height': DIM, 'width': DIM, 'id': fileID})
        with open(str(truthFile)) as f:
            lines = [line for line in f.read().split('\n') if line]
        for line in lines:
            objID, x, y, w, h = line.split(' ')
            objID = int(objID)
            x = int(float(x) * DIM)
            y = int(float(y) * DIM)
            w = int(float(w) * DIM)
            h = int(float(h) * DIM)
            dataset['annotations'].append({'id': len(dataset['annotations']), 'bbox': [x, y, w, h], 'image_id': fileID, 'segmentation': [], 'ignore': 0, 'area': w*h, 'iscrowd': 0, 'category_id': objID})
    return dataset

def results2json(path):
    results = []
    for resFile in path.glob('*.txt'):
        if not re.findall(r'\d+\.', str(resFile)):
            continue
        fileID = int(re.findall(r'\d+\.', str(resFile))[0][:-1])
        with open(str(resFile)) as f:
            lines = [line for line in f.read().split('\n') if line]
        for line in lines:
            objID, x, y, w, h = line.split(' ')
            objID = int(objID)
            x = int(float(x) * DIM)
            y = int(float(y) * DIM)
            w = int(float(w) * DIM)
            h = int(float(h) * DIM)
            results.append({'image_id': fileID, 'category_id': objID, 'bbox': [x, y, w, h], 'score': 1})
    return results

import json
with open(str(truthPath)+'/dataset.json', 'w') as f:
    f.write(json.dumps(dataset2json(truthPath)))

with open(str(detectedPath)+'/results.json', 'w') as f:
    f.write(json.dumps(results2json(detectedPath)))

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

annFile = str(truthPath)+'/dataset.json'
resFile = str(detectedPath)+'/results.json'

cocoGt=COCO(annFile)

cocoDt=cocoGt.loadRes(resFile)

annType = 'bbox'
cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
