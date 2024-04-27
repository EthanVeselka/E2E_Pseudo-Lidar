#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE            #Do not propagate environment
#SBATCH --get-user-env=L         #Replicate login environment
#
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=pseudo_lidar_pipeline       #Set the job name to "JobName"
#SBATCH --time=2:00:00           #Set the wall clock limit to 2hr and 00min
#SBATCH --ntasks=1               #Request tasks/cores per node
#SBATCH --mem=8G                 #Request 8GB per node 
#SBATCH --output=output.%j       #Send stdout/err to "output.[jobID]" 
#SBATCH --gres=gpu:1                 #Request GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue
#SBATCH --gres=gpu:a100:1        #(N is either 1 or 2)
#
##OPTIONAL JOB SPECIFICATIONS
##SBATCH --mail-type=ALL              #Send email on all job events
##SBATCH --mail-user=email_address    #Send all emails to email_address 
#
##First Executable Line
#
module load GCC/11.3.0
module load CUDA/11.7.0
module load OpenMPI/4.1.4
module load TensorFlow/2.11.0-CUDA-11.7.0
module load PyTorch/1.12.0-CUDA-11.7.0
module load OpenCV/4.6.0-contrib
module load scikit-learn/1.1.2
pip3 install torchvision==0.2.0

# nvidia-driver/default

# cd processing/
# echo Began processing
# python3 process.py --clean 
# python3 process.py --sample
# python3 process.py --gen_disp 
# echo Finished processing
# cd ../

pip3 list | grep torchvision
cd models/PSMNet/scripts/
# echo Began training
# sh train.sh
# echo Finished training
echo Began predict
sh predict.sh
echo Finished predict
# cd ../../../

# echo Started generating pseudo lidar
# cd processing/pseudo_lidar/
# python3 generate_pl.py --use_pred
# python3 generate_pl.py
# echo Finished generating pseudo lidar
# cd ../../

# module purge

# cd models/FPN/

# module load CUDA/8.0.44
# module load cuDNN/6.0.21-CUDA-8.0.44

# # Load Python 3.6.4
# export PATH=/sw/eb/sw/Python/3.6.4-golf-2018a/bin:$PATH
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/eb/sw/Python/3.6.4-golf-2018a/lib
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/eb/sw/Python/3.6.4-golf-2018a/lib/python3.6/site-packages/numpy-1.14.0-py3.6-linux-x86_64.egg/numpy/core/lib

# # Load GCC 5.3
# export PATH=$(pwd)/venv/dependencies/gcc/usr/local/bin:$PATH
# export LD_LIBRARY_PATH=$(pwd)/venv/dependencies/gcc/usr/local/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$(pwd)/venv/dependencies/gcc/usr/local/lib64:$LD_LIBRARY_PATH

# source venv/bin/activate

# cd scripts/

# # sh fpn_requirements.sh
# echo Starting prep data
# sh command_prep_data.sh
# echo Finished prep data
# echo Starting train
# sh command_train_v2.sh
# echo Starting test
# sh command_test_v2.sh

# deactivate

# unset PATH
# unset LD_LIBRARY_PATH

# export PATH=/usr/lib64/qt-3.3/bin:/sw/local/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/usr/lpp/mmfs/bin