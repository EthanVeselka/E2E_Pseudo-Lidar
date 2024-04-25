python3 ../finetune_3d.py \
    --loadmodel ../saved_models/pretrained_sceneflow.tar \
    --savemodel ../saved_models/finetune_pre \
    --datapath carla_data/data/ \
    --split_file carla_data/output \
    --maxdisp 192 \
    --lr_scale 50 \
    --epochs 10 \
    --btrain 4 \
    --cuda