FOLDER=/var/tmp/zoom-datasets/09-12-2100/merged

find $FOLDER -name *.lh0 -delete
find $FOLDER -name *.corrs -delete
find $FOLDER -name *.lh0+ -delete
find $FOLDER -name *.svg -delete
find $FOLDER -name *.uv -delete
find $FOLDER -name *.gp -delete
find $FOLDER -name *.intrinsics -delete
find $FOLDER -name intrinsics.samples -delete
find $FOLDER -name *.model -delete

echo ''
echo -e '\033[1;33m//     Update code       //'
echo -e '\033[0m'
rsync -az --ignore-times --delete --exclude pipeline.sh rpradeep@rpradeep.webhop.net:studio/zoomcalib/ .

echo ''
echo -e '\033[1;33m//  Homography at center  //'
echo -e '\033[0m'
parallel -X ./homography_at_center.py ::: $FOLDER/*/pose*.png

echo ''
echo -e '\033[1;33m//  Refine homographies   //'
echo -e '\033[0m'
./refine_homographies2.py $FOLDER

echo ''
echo -e '\033[1;33m//  Refine homography subsets   //'
echo -e '\033[0m'
./refine_homography_subsets.py $FOLDER

echo ''
echo -e '\033[1;33m//  Intrinsics Model   //'
echo -e '\033[0m'
./zoom_intrinsics_model.py $FOLDER

# echo ''
# echo -e '\033[1;33m//  Refine extrinsics   //'
# echo -e '\033[0m'
# ./refine_extrinsics.py $FOLDER

echo ''
echo -e '\033[1;33m//  Create Visualizations  //'
echo -e '\033[0m'
parallel -X python visualize_distortion.py ::: $FOLDER/*/pose0.lh0+

# echo ''
# echo -e '\033[1;33m//     Build zoom model    //'
# echo -e '\033[0m'
# ./papers/icra16/scripts/zoom_intrinsics_plot.py $FOLDER
