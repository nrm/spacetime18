#!/bin/bash

taskid=$1
host=$(hostname -s)
result="result_${taskid}_${host}.csv"


cd ~/project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# source ~/venv/bin/activate

l="/opt/spacetime/layoutslayout_2021-06-15.tif"
for c in $(ls /opt/spacetime/1_20/*_0000.tif):
do
        echo "L is $l; C is $c";
        python test.py --layout ${l} --crop ${c} --taskid ${result}
done

# cp result.csv ~/
cp ${result} ~/
echo "End"
