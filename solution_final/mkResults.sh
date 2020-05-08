#!/bin/bash
echo "kernel, objects-clutter, lbp, test, iou, router, nerf, glasses, vader, brush, pig, cardboard, stand, fuse, catan, precision, recall"

for file in `ls tests/*_32bins_split*`; do
	kernel=`cut -d'/' -f2 <<< $file | cut -d '_' -f1`
	lbp=`cut -d'/' -f2 <<< $file | cut -d '_' -f3`
	tes=`cut -d'/' -f2 <<< $file | rev | cut -d '_' -f1 | rev | cut -d'-' -f2 | cut -d'.' -f1`
	iou=`tail -n 1 $file | awk '{print $3}'`
	precision=`grep "Precision.*IoU=0.50 " $file | rev | awk '{print $1}' | rev`
	recall=`grep "Recall.* all.*maxDets=100" $file | rev | awk '{print $1}' | rev`
	objects=`cut -d'-' -f1 <<< $file | rev | cut -d'_' -f2 | rev`
	clutter=`cut -d'-' -f1 <<< $file | rev | cut -d'_' -f1 | rev`
	if [ $objects != "mobi" ] && [ $objects != "c615" ]; then
		objects="all"
		clutter="all"
	fi
	echo -n "$kernel, $objects-$clutter, $lbp, $tes, $iou, "
	for obj in "router" "nerf" "glasses" "vader" "brush" "pig" "cardboard" "stand" "fuse" "catan"; do
		echo -n "`grep $obj $file | awk '{print $5}'`, "
	done
	echo "$precision, $recall"
done
