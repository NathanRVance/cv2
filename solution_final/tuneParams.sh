#!/bin/bash

DIR=/tmp/clutter
NUMTRAIN=70
NUMVAL=30
NUMTOT=$((NUMTRAIN+NUMVAL))
rm -rf $DIR
../clutterizer/clutterize.py -s $DIR/train/generated.jpg -n $NUMTOT
./labelme2coco.py $DIR/train $DIR/train

mkdir $DIR/validate
for num in `seq $NUMTRAIN $((NUMTOT-1))`; do
    mv $DIR/train/generated$num.* $DIR/validate
done
cp $DIR/train/names.txt $DIR/validate
for cutoff in .14 .15 .16 .17 .18; do
    ./solution.py $DIR/train $DIR/validate $DIR/answers --cutoff $cutoff
    ./evaluate.py $DIR/answers $DIR/validate | tee $DIR/validate$cutoff.txt
    rm -r $DIR/answers
done
#for lbp in 0 1 2 3 4; do
#    ./solution.py $DIR/train $DIR/validate $DIR/answers --lbp $lbp
#    ./evaluate.py $DIR/answers $DIR/validate | tee $DIR/validate_lbp$lbp.txt
#    rm -r $DIR/answers
#done
