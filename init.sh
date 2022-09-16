#!/bin/bash

echo Setting up environment
module load python/3.8
module load httpproxy

pwd

virtualenv ./tensorflow-test
source ./tensorflow-test/bin/activate 
pwd

echo Installing TF
pip install --no-index comet-ml tensorflow numpy matplotlib pillow tqdm pandas


echo installed!
python comet-test.py

echo DONE!