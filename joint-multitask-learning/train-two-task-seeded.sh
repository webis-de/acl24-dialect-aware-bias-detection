#! /bin/bash

set -e

TIMESTAMP=$(date +"%Y%m%d%H%M%S")
SEEDS=(23 42 271 314 1337)
SBIC_TASKS=("groupYN" "intentYN" "lewdYN" "offensiveYN" "ingroupYN")

for seed in "${SEEDS[@]}"; do
    {
        echo -e "\e[31m==================================================\e[0m"
        echo -e "\e[31mStarting finetuning for seed ${seed}\e[0m"
        echo -e "\e[31m==================================================\e[0m"

        for task in "${SBIC_TASKS[@]}"; do
            {
                echo -e "\e[31m==================================================\e[0m"
                echo -e "\e[31mStarting two-task MTL training for task ${task}\e[0m"
                echo -e "\e[31m==================================================\e[0m"

                python train.py \
                    --data_dir "./intermediate" \
                    --base_model "./model/roberta-base" \
                    --cache_path "/tmp" \
                    --model_identifier "two-task-mtl_${task}_${TIMESTAMP}" \
                    --seed "${seed}" \
                    --labels aae_dialect ${task}
            } || exit 1
        done

        echo ""
        echo ""
    } || exit 1
done
