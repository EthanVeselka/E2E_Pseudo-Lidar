python ../finetune_3d.py \
    --maxdisp 192 \
    --model stackhourglass \
    --datapath carla_data/example_data/ \
    --split_file carla_data/output \
    --epochs 300 \
    --lr_scale 50 \
    --savemodel ./saved_models/ \
    --btrain 12