#!/bin/bash

DIR=/tmp/clutter
NUM=2
rm -rf $DIR
../clutterizer/clutterize.py -s $DIR/data/generated.jpg -n $NUM
./labelme2coco.py $DIR/data $DIR/coco
./solution.py $DIR/data $DIR/answers
./evaluate.py $DIR/answers $DIR/coco
