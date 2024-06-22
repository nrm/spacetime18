#!/bin/bash

taskid=$1
host=$(hostname -s)
result="result_${taskid}_${host}.csv"


cd ~/project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# source ~/venv/bin/activate

l="/opt/spacetime/layouts/layout_2021-06-15.tif"
# for c in $(ls /opt/spacetime/1_20/*_0000.tif):
files=(
    "/opt/spacetime/1_20/crop_0_0_0000.tif"
    "/opt/spacetime/1_20/crop_0_1_0000.tif"
    "/opt/spacetime/1_20/crop_0_2_0000.tif"
    "/opt/spacetime/1_20/crop_0_3_0000.tif"
    "/opt/spacetime/1_20/crop_1_0_0000.tif"
    "/opt/spacetime/1_20/crop_1_1_0000.tif"
    "/opt/spacetime/1_20/crop_1_2_0000.tif"
    "/opt/spacetime/1_20/crop_1_3_0000.tif"
    "/opt/spacetime/1_20/crop_2_0_0000.tif"
    "/opt/spacetime/1_20/crop_2_1_0000.tif"
)

for c in "${files[@]}"
do
        echo "L is $l; C is $c";
        python test.py --layout ${l} --crop ${c} --taskid ${result}
done

# cp result.csv ~/
cp ${result} ~/
echo "End"
