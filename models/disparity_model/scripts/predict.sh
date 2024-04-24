python3 ../predict.py \
    --datapath carla_data/data/ \
    --split_file carla_data/output \
    --loadmodel ../saved_models/finetune_pre/finetune_5.tar \
    --save_figure \
    --cuda \
    # --test_accuracy
    # --loadmodel ../saved_models/pretrained_sceneflow.tar \
    # --loadmodel ../saved_models/finetune_pre/finetune_5.tar \
# Default : Predicts only on test set, producing disparity maps and saving locally to ../predictions unless --all is specified
# --all : Predicts disparity maps for all frames in splits, saves to respective frame's /output