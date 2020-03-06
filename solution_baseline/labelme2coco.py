#!/usr/bin/env python3

import os
import glob
import json
from collections import OrderedDict

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('IN_DIR', help='directory containing json input files')
parser.add_argument('OUT_DIR', help='directory to record COCO formatted files')
args = parser.parse_args()

LABELS_DIR = args.OUT_DIR
JSON_DIR = args.IN_DIR

if not os.path.exists(LABELS_DIR):
    os.makedirs(LABELS_DIR)

train_image_names = []
valid_image_names = []
class_names     = OrderedDict()
new_annotation  = dict()

for json_name in glob.glob(JSON_DIR + "/*.json"):

    with open(json_name, "r") as fp:
        data = json.load(fp)

    # read json annotation
    image_path      = data['imagePath']
    image_w         = data['imageWidth']
    image_h         = data['imageHeight']

    new_annotation[image_path] = []

    for s in data['shapes']:

        # associate a number to the new class name
        label = s['label']
        if label not in class_names:
            class_names[label] = len(class_names)

        # extract bounding box coordinates
        px = [ x for x, _ in s['points'] ]
        py = [ y for _, y in s['points'] ]

        min_x, min_y = min(px), min(py)
        max_x, max_y = max(px), max(py)

        # convert pixel coordinates to [0, 1] coordinate system
        w = (max_x - min_x) / image_w
        h = (max_y - min_y) / image_h

        # Remove zero-sized regions
        if w == 0 or h == 0:
            continue

        xc = ( max_x + min_x ) / (2 * image_w)
        yc = ( max_y + min_y ) / (2 * image_h)

        # sanity check
        assert w <= image_w and h <= image_h and \
                xc < image_w and yc < image_h, \
                "In {} -> '{}' annotations are out of boundaries".\
                format(image_path, s['label'])

        new_line = [str(class_names[label]), str(xc), str(yc), str(w), str(h)]
        new_annotation[image_path].append(new_line)


    # store new labels in files
    for img, annotations in new_annotation.items():

        # replace file extension '.jpg' for '.txt'
        img = ".".join(img.split('.')[:-1])
        txt_path = os.path.join(LABELS_DIR, img + ".txt")

        # write annotation to text file
        new_lines = ""
        for ann in annotations:
            new_lines += " ".join(ann) + "\n"
        # print(new_lines)

        with open(txt_path, "w+") as fp:
            fp.write(new_lines)


# store sorted class names in './data/yolo.names':
sorted_classes = sorted(class_names.items(), key=lambda x: x[1])
output = ""
print("Classes:")
for c, i in sorted_classes:
    print(i, c)
    output += "{}\n".format(c)

with open(LABELS_DIR+"/names.txt", "w") as fp:
    fp.write(output)
with open(JSON_DIR+"/names.txt", "w") as fp:
    fp.write(output)

# store metadata in './data/yolo.data':

#yolo_data = """
#classes = {}
#train   = data/train.txt
#valid   = data/valid.txt
#names   = data/yolo.names
#backup  = backup
#""".format(len(class_names))

#with open("./data/yolo.data", "w") as fp:
#    fp.write(yolo_data)

