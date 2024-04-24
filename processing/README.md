
# Processing Scripts

## Process

`process.sh`
- Description : Processing script, specifies processing options outlined below, runs `process.py`.

**--clean** | Cleans frame/output folders (disp/pl/pred_disp), runs `utils.clean_output`.

**--gen_disp** | Generates disparities for all frames, saves to frame/output, runs `pseudo_lidar/generate_disp.py`.

**--sample** | Generates train/val/test splits, saves listfiles to carla_data/output, runs `sample.py`.

`process.grace_job`
- Description : Loads required modules in grace, runs `process.sh`.

## Pseudo-Lidar

`gen_pl.sh`
- Description : Generates pseudo_lidar using ground truth | predicted disparities, runs `pseudo_lidar/generate_pl.py`.

**--use_pred** | Generate Pseudo-Lidar from predicted disparities [default: False] (Requires PSMNet predicted disparities).

**--root_dir** | Root data directory [default: carla_data/data].

**--max_high** | Max height of accepted points for Pseudo_Lidar.

**--is_depth** | Generates PL from depth maps, rather than disparities.

**--calib_dir**- | Calibration matrix directory [default: carla_data/data].

**--listfile_dir** | train/val/test split file directory [default: carla_data/output].