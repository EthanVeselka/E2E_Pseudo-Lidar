python3 ../predict.py \
    --datapath carla_data/data/ \
    --split_file carla_data/output \
    --loadmodel ../saved_models/finetune_pre/finetune_8.tar \
    --save_figure \
    --cuda \
    --test_accuracy
    # --loadmodel ../saved_models/pretrained_sceneflow.tar \
    # --loadmodel ../saved_models/finetune_pre/finetune_5.tar \