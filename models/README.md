Disparity map estimation (PSMNet) and 3D object detection/bounding box prediction using Frustum PointNet (FPN)

We provide scripts for setting up environments and installing dependencies for the implemented models below.

---
# PSMNet
PSMNet implements the Stackhourglass CNN model that performs estimation of the disparity between a left and right 
image pair for a frame, using a disparity map generated from the ground truth LiDAR for the frame as the target.
PSMNet requirements can be found in `grace_psm_train.grace_job`. Uses PyTorch 1.12 and CUDA 11.7

## Metrics 
- Average loss
- Average per epoch 3-px error from left disparity

## Preprocessing
`process.sh`
- Description : Processing script, specifies processing options outlined below, runs `process.py`.

*Args*:

**--clean** | Cleans frame/output folders (disp/pl/pred_disp).

**--gen_disp** | Generates disparities for all frames, saves to frame/output.

**--sample** | Generates train/val/test splits, saves listfiles to carla_data/output.

## Training
Requires train/val/test split files, reports the batch and average per epoch 3-px validation error.

`train.sh`
- Description : Training script, specifies training parameters outlined below, runs `finetune_3d.py`.

### Parameters:
**--cuda** | Use CUDA.

**--maxdisp** | Maximum disparity.

**--epochs** | Number of training epochs.

**--btrain** | Number of training batches.

**--loadmodel** | Path of pretrained model to load.

**--savemodel** | Save path for trained model.

**--start_epoch** | Starting epoch of loaded model.

**--datapath** | Starting epoch of loaded model.

**--split_file** | Path to train/val/test split files.

**--seed** | Initialization seed.

## Prediction
Require val/test split files, reports average per epoch 3-px test error if **--test_accuracy** is specified.

`predict.sh`
- Description : Testing script, specifies eval parameters outlined below, runs `predict.py`.
- Default : Predicts only on test set, producing disparity maps in .npy format and saving locally to ../predictions unless --all is specified.

### Parameters:
**--cuda** | Use CUDA.

**--maxdisp** | Maximum disparity.

**--save_figure** | Save disparity map as png for viewing, defaults to saving as .npy unless True.

**--loadmodel** | Path of pretrained model to load.

**--test_accuracy** | Reports test accuracy instead of saving disparity maps.

**--all** | Predicts on all frames in all splits, producing disparity maps and saving to respective frame folders.

**--datapath** | Starting epoch of loaded model.

**--split_file** | Path to train/val/test split files.

**--seed** | Initialization seed.

---
# Frustum PointNet
Frustum PointNet implements the model that performs 3D object detection and boundinig box prediction for a frame, 
using LiDAR and ground truth bounding boxes. Modify lidar_filename option in 
FPNDataset.get_lidar() to use Pseudo-Lidar or LiDAR accordingly.

See [FPN requirements](#req) for environment setup.

## Metrics 
- Average loss
- Average segmentation accuracy
- Box IoU (ground/3D)
- Box estimation accuracy (IoU=0.7)

## Preprocessing
`command_prep_data.sh`
- Description : Training script, specifies training parameters outlined below, runs `prepare_data.py`.

### Parameters:
**--all** | Use all classes.

**--car_only** | Use only Cars.

**--carpedcyc_only** | Use Cars, Pedestrians, Bicycles.

**--gen_train** | Generate FPN training .pickle data.

**--gen_val** | Generate FPN validation .pickle data.

**--demo** | Visualize LiDAR and bboxes, requires mayavi.

## Training
Requires train/val/test split files, reports .

`command_train_v2.sh`
- Description : Training script, specifies training parameters outlined below, runs `train.py`.

### Parameters:
**--gpu** | GPU device to use.

**--model** | Model name [default: frustum_pointnets_v2].

**--restore_model_path** | Restore model path e.g. log/model.ckpt [default: None].

**--log_dir** | Log dir [default: log_v2].

**--num_point** | Point Number [default: 2048].

**--max_epoch** | Epoch to run [default: 201].

**--batch_size** | Batch Size during training [default: 32].

**--learning_rate** | Initial learning rate [default: 0.001].

**--momentum** | Initial learning rate [default: 0.9].

**--optimizer** | adam or momentum [default: adam].

**--decay_step** | Decay step for lr decay [default: 200000].

**--decay_rate** | Decay rate for lr decay [default: 0.7].

**--no_intensity** | Only use XYZ for training.

## Prediction
Requires train/val/test split files, evaluates specified FPN model

`command_test_v2.sh`
- Description : Testing script, specifies eval parameters outlined below and runs `test.py`.

### Parameters
**--gpu** | GPU device to use.

**--model** | Model name [default: frustum_pointnets_v2].

**--model_path** | Model checkpoint file path [default: log_v2/model.ckpt].

**--num_point** | Point Number [default: 2048].

**--batch_size** | Batch Size during training [default: 32].

**--output** | Output file/folder name [default: test_results].

**--dump_result** | If true, also dump results to .pickle file.

**--data_path** | Frustum dataset pickle filepath [default: None].

## FPN Requirements {#req}
FPN has various requirements for environment setup. It requires TensorFlow==1.4 and CUDA 8.0 to compile; we trained using Texas A&M University HPRC Grace, and have provided scripts for Grace environment setup and job submission that handles installing/loading the required libraries.

`create_FPN_venv.sh` 
- Description : Creates the FPN virtual environment on Grace, manually compiles glibc v2.23 (This may take up to an hour). 

`fpn_pip_requirements.txt` 
- Description : List of pip dependencies. 

`fpn_venv.grace_job` 
- Description : Activates virtual environment, trains | tests FPN on Grace. 

### Compiling tf_ops
The FPN model depends on precompiled tf binaries from models/tf_ops

`fpn_compile.sh`
- Description : Compiles tf_ops dependencies, requires TensorFlow 1.4

