#! /bin/bash

TIMESTAMP=$(date +"%Y%m%d%H%M%S")

accelerate launch --config_file accelerate_config.yaml train.py \
    --train_data "../data/sbic-data/sbic-train-prep.csv" \
    --val_data "../data/sbic-data/sbic-val-prep.csv" \
    --test_data "../data/sbic-data/sbic-test-prep.csv" \
    --base_model "./model/bert-base-uncased" \
    --cache_path "/tmp" \
    --model_identifier "finetune_${TIMESTAMP}"
