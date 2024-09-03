#! /bin/bash

set -e

TIMESTAMP=$(date +"%Y%m%d%H%M%S")
SEEDS=(23 42 271 314 1337)

for seed in "${SEEDS[@]}"; do
    {
        echo -e "\e[31m==================================================\e[0m"
        echo -e "\e[31mStarting finetuning for seed ${seed}\e[0m"
        echo -e "\e[31m==================================================\e[0m"

        python train.py \
            --data_dir "./intermediate" \
            --base_model "./model/deberta-v3-base" \
            --cache_path "/tmp" \
            --model_identifier "joint-mtl-no-aae_${TIMESTAMP}" \
            --seed "${seed}" \
            --labels groupYN intentYN lewdYN offensiveYN ingroupYN

        echo ""
        echo ""
    } || exit 1
done
