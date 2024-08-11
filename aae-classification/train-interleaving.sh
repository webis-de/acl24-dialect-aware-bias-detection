#! /bin/bash

set -e

SEEDS=(23 42 271 314 1337)

for seed in "${SEEDS[@]}"; do
    {
        echo -e "\e[31m==================================================\e[0m"
        echo -e "\e[31mStarting finetuning for seed ${seed}\e[0m"
        echo -e "\e[31m==================================================\e[0m"

        accelerate launch --config_file accelerate_config.yaml train-interleaving.py \
            --seed "${seed}"

        echo ""
        echo ""
    } || exit 1
done
