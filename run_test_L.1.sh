#!/bin/bash

# taskid=$1
result=$1
layout_file=$2
# host=$(hostname -s)
# result="result_${taskid}_${host}.csv"
# result="result_${taskid}.csv"

# Проверяем, существует ли файл layout_file
if [ ! -f "$layout_file" ]; then
    echo "Error: Layout file $layout_file does not exist."
    exit 1
fi

cd ~/project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

files=(
    "/opt/spacetime/1_20/crop_0_0_0000.tif"
    # "/opt/spacetime/1_20/crop_0_1_0000.tif"
    # "/opt/spacetime/1_20/crop_0_2_0000.tif"
    # "/opt/spacetime/1_20/crop_0_3_0000.tif"
    # "/opt/spacetime/1_20/crop_1_0_0000.tif"
    # "/opt/spacetime/1_20/crop_1_1_0000.tif"
    # "/opt/spacetime/1_20/crop_1_2_0000.tif"
    # "/opt/spacetime/1_20/crop_1_3_0000.tif"
    # "/opt/spacetime/1_20/crop_2_0_0000.tif"
    # "/opt/spacetime/1_20/crop_2_1_0000.tif"
)

for crop_file in "${files[@]}"
do
    echo "Layout is $layout_file; Crop is $crop_file"
    python test.py --layout "$layout_file" --crop "$crop_file" --taskid "$result"
done

cp "$result" ~/
echo "End"
