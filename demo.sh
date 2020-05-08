#!/bin/bash

echo "This is the demo of the CV2 code"

echo "We will start walking through all of the major components"

echo "Now generating 100 synthetic images, and saving them to /tmp/clutter/train"

./clutterizer/clutterize.py -s /tmp/clutter/train/generated.jpg -n 100
./solution_final
