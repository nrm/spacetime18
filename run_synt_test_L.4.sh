#!/bin/bash

# taskid=$1
result=$1
layout_file=$2
project_dir=$3

# Проверяем, существует ли файл layout_file
if [ ! -f "$layout_file" ]; then
    echo "Error: Layout file $layout_file does not exist."
    exit 1
fi

cd ~/${project_dir}
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

files=(
    "/opt/spacetime/synt_tiles/crop_3_6_0000.tif"
    # "/opt/spacetime/synt_tiles/crop_3_7_0000.tif"
    # "/opt/spacetime/synt_tiles/crop_4_0_0000.tif"
    # "/opt/spacetime/synt_tiles/crop_4_1_0000.tif"
    # "/opt/spacetime/synt_tiles/crop_4_2_0000.tif"
    # "/opt/spacetime/synt_tiles/crop_4_3_0000.tif"
    # "/opt/spacetime/synt_tiles/crop_4_4_0000.tif"
    # "/opt/spacetime/synt_tiles/crop_4_5_0000.tif"
    # "/opt/spacetime/synt_tiles/crop_4_6_0000.tif"
    # "/opt/spacetime/synt_tiles/crop_4_7_0000.tif"
)

for crop_file in "${files[@]}"
do
    echo "Layout is $layout_file; Crop is $crop_file"
    python test.py --layout "$layout_file" --crop "$crop_file" --taskid "$result"
done

cp "$result" ~/
echo "End"
