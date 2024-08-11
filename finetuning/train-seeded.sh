#! /bin/bash

set -e

TIMESTAMP=$(date +"%Y%m%d%H%M%S")
SEEDS=(23 42 271 314 1337)

for seed in "${SEEDS[@]}"; do
    {
        echo -e "\e[31m==================================================\e[0m"
        echo -e "\e[31mStarting finetuning for seed ${seed}\e[0m"
        echo -e "\e[31m==================================================\e[0m"

        accelerate launch --config_file accelerate_config.yaml train.py \
            --train_data "../data/sbic-data/sbic-train-prep.csv" \
            --val_data "../data/sbic-data/sbic-val-prep.csv" \
            --test_data "../data/sbic-data/sbic-test-prep.csv" \
            --base_model "./model/roberta-base" \
            --cache_path "/tmp" \
            --model_identifier "finetune_${TIMESTAMP}" \
            --seed "${seed}"

        echo ""
        echo ""
    } || exit 1
done
