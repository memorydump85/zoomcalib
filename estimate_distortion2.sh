#! /bin/bash

FOLDER=$1

parallel -X ./homography_at_center.py ::: $FOLDER/pose?/*.png
echo ''
echo ''
./refine_homographies.py $FOLDER pose0
echo ''
echo ''
parallel -X python visualize_distortion.py ::: $FOLDER/pose0/*.lh0+
echo ''
echo ''
./zoom_model.py $FOLDER/pose0/*.uv