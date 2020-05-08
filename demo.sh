#!/bin/bash

DIR=/tmp/clutter
NUMTRAIN=70
NUMVAL=1
NUMTOT=$((NUMTRAIN+NUMVAL))
rm -rf $DIR

echo "This is the demo of the CV2 code"

echo "We will start walking through all of the major components"

echo "Now generating $NUMTOT synthetic images, and saving them to $DIR/train"
./clutterizer/clutterize.py -s $DIR/train/generated.jpg -n $NUMTOT
./solution_final/labelme2coco.py $DIR/train $DIR/train

mkdir $DIR/test
for num in `seq $NUMTRAIN $((NUMTOT-1))`; do
    mv $DIR/train/generated$num.* $DIR/test
done
cp $DIR/train/names.txt $DIR/test

clutterDir=`grep 'clutter:' conf.yaml | awk '{print $2}'`
for file in `find $clutterDir | grep json`; do
    jpg="`rev <<< $file | cut -d'.' -f2 | rev`.jpg"
    cp $file $DIR/test
    cp $jpg $DIR/test
    break
done
./solution_final/labelme2coco.py $DIR/test $DIR/test

echo "Now calculating bounding boxes for $NUMVAL validation images and 1 test image"
./solution_final/solution.py --parallel False $DIR/train $DIR/test $DIR/answers
./solution_final/evaluate.py $DIR/answers $DIR/test

echo "Now displaying the images. Press any key to continue."
./solution_final/visualize.py $DIR/answers $DIR/test
