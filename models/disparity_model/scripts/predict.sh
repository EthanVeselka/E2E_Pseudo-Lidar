python3 ../predict.py \
    --cuda \
    --datapath carla_data/example_data/ \
    --split_file carla_data/output \
    --loadmodel ../saved_models/finetune_3.tar \
    --save_figure