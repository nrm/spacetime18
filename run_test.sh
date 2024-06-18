#!/bin/bash

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python test.py --layout /opt/spacetime/layouts/layout_2021-08-16.tif --crop /opt/1_20/crop_0_0_0000.tif
cp result.csv ~/
