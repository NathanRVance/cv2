#!/bin/bash

name=$1
echo "Running tests $name"
lbp=$2
echo "Using lbp setting $lbp"


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
./solution.py --lbp $lbp $DIR/train $DIR/validate $DIR/answers
./evaluate.py $DIR/answers $DIR/validate | tee tests/$name-validate.txt

mkdir $DIR/test
clutterDir=`grep 'clutter:' conf.yaml | awk '{print $2}'`
for file in `find $clutterDir | grep json`; do
    jpg="`cut -d'.' -f1 <<< $file`.jpg"
    cp $file $DIR/test
    cp $jpg $DIR/test
done
cp $DIR/train/names.txt $DIR/test
./labelme2coco.py $DIR/test $DIR/test
./solution.py --lbp $lbp $DIR/train $DIR/test $DIR/testAnswers
./evaluate.py $DIR/testAnswers $DIR/test | tee tests/$name-test.txt
