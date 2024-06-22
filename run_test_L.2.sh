#!/bin/bash

# taskid=$1
# host=$(hostname -s)
# result="result_${taskid}_${host}.csv"
# result="result_${taskid}.csv"
result=$1
layout_file=$2

# Проверяем, существует ли файл layout_file
if [ ! -f "$layout_file" ]; then
    echo "Error: Layout file $layout_file does not exist."
    exit 1
fi

cd ~/project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# source ~/venv/bin/activate

# l="/opt/spacetime/layouts/layout_2021-08-16.tif"
# l="/opt/spacetime/layouts/layout_2021-06-15.tif"
files=(
    "/opt/spacetime/1_20/crop_2_2_0000.tif"
    "/opt/spacetime/1_20/crop_2_3_0000.tif"
    "/opt/spacetime/1_20/crop_3_0_0000.tif"
    "/opt/spacetime/1_20/crop_3_1_0000.tif"
    "/opt/spacetime/1_20/crop_3_2_0000.tif"
)

for crop_file in "${files[@]}"
do
        # echo "L is $l; C is $c";
        # python test.py --layout ${l} --crop ${c} --taskid ${result}
    echo "Layout is $layout_file; Crop is $crop_file"
    python test.py --layout "$layout_file" --crop "$crop_file" --taskid "$result"
done

# cp result.csv ~/
cp ${result} ~/
echo "End"
