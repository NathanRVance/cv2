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
from shapely.geometry import Polygon

DIM = 1000

def iou(path1, path2):
    ious = {}
    dat1 = {}
    dat2 = {}
    for path, dat in [(path1, dat1), (path2, dat2)]:
        for resFile in path.glob('*.txt'):
            if not re.findall(r'\d+\.', str(resFile)):
                continue
            fname = str(resFile).split('/')[-1]
            dat[fname] = {}
            with open(str(resFile)) as f:
                for line in f.read().split('\n'):
                    if not line:
                        continue
                    ID, xc, yc, w, h = [float(part) for part in line.split(' ')]
                    ID = int(ID)
                    p = Polygon([(xc-w/2,yc-h/2), (xc+w/2, yc-h/2), (xc+w/2, yc+h/2), (xc-w/2, yc+h/2)])
                    if ID not in dat[fname]:
                        dat[fname][ID] = p
                    else:
                        dat[fname][ID] = p.union(dat[fname][ID])
    for fname in set(dat1.keys()).intersection(dat2.keys()):
        for ID in set(dat1[fname].keys()).union(dat2[fname].keys()):
            if ID in dat1[fname]:
                p1 = dat1[fname][ID]
            else:
                p1 = Polygon()
            if ID in dat2[fname]:
                p2 = dat2[fname][ID]
            else:
                p2 = Polygon()
            if ID not in ious:
                ious[ID] = [0, 0]
            ious[ID][0] += p1.intersection(p2).area
            ious[ID][1] += p1.union(p2).area

    for ID in sorted(ious.keys()):
        if ious[ID][1] == 0:
            print('Object {} wasn\'t there nor was it detected.'.format(mappings[ID]))
        else:
            print('IOU for object {}: {}'.format(mappings[ID], ious[ID][0] / ious[ID][1]))
    print('Total IOU: {}'.format(sum(ious[ID][0] for ID in ious) / sum(ious[ID][1] for ID in ious)))

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
iou(detectedPath, truthPath)
