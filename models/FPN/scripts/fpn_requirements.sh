
# Create new conda env with reqs
conda create --name e2e_pl python=3.6
conda activate e2e_pl
python -m pip install tensorflow==1.4
python -m pip install opencv-python
python -m pip install pillow
python -m pip install scipy

# Compile tf_ops
sh ../models/tf_ops/3d_interpolation/tf_interpolate_compile.sh
sh ../models/tf_ops/grouping/tf_grouping_compile.sh
sh ../models/tf_ops/sampling/tf_sampling_compile.sh
