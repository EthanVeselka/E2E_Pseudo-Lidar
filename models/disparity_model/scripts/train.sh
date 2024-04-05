python3 ../finetune_3d.py \
    --cuda \
    --maxdisp 192 \
    --datapath carla_data/example_data/ \
    --split_file carla_data/output \
    --epochs 3 \
    --lr_scale 50 \
    --btrain 2