
# Create new conda env with reqs
# conda create --name e2e_pl python=3.6
# conda activate e2e_pl
# conda config --add channels conda-forge
# conda install tensorflow==1.4
# conda install opencv-python
# conda install pillow
# conda install scipy

# Compile tf_ops
cd ../models/tf_ops/3d_interpolation
sh tf_interpolate_compile.sh
cd ../grouping
sh tf_grouping_compile.sh
cd ../sampling
sh tf_sampling_compile.sh
