# Compile tf_ops
cd ../models/tf_ops/3d_interpolation
sh tf_interpolate_compile.sh
cd ../grouping
sh tf_grouping_compile.sh
cd ../sampling
sh tf_sampling_compile.sh
