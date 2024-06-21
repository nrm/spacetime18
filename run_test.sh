#!/bin/bash

cd ~/project
# python3 -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt
source ~/venv/bin/activate

for l in $(ls /opt/spacetime/layouts/):
do
        for c in $(ls ~/spacetime18/1_20/*_0000.tif):
        do
                echo "L is $l; C is $c";
                python test.py --layout ${l} --crop ${c}
        done
done

# python test.py --layout /opt/spacetime/layouts/layout_2021-08-16.tif --crop /opt/1_20/crop_0_0_0000.tif
cp result.csv ~/
echo "End"
