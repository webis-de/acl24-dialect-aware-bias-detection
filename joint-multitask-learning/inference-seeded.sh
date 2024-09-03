#! /bin/bash

SEEDS=(23 42 271 314 1337)
BASE_MODEL_NAME="deberta-v3-base"
MODEL_PATH="./model/__final/multi-seed-runs"

for seed in "${SEEDS[@]}"; do
{
    echo -e "\e[31m==================================================\e[0m"
    echo -e "\e[31mStarting inference for seed ${seed}\e[0m"
    echo -e "\e[31m==================================================\e[0m"

    # Find the latest trained model for the current task
    LATEST_MODEL=$(ls "${MODEL_PATH}" | grep "^${BASE_MODEL_NAME}-joint-mtl_[0-9]\+-seed${seed}$" | sort -t '_' -k 3 -n | tail -n 1)

    echo -e "\e[31mChose model '${LATEST_MODEL}'.\e[0m"

    echo -e "\e[31m==================================================\e[0m"
    echo ""

    python inference.py \
        --data_dir "./intermediate" \
        --base_model "./model/${BASE_MODEL_NAME}" \
        --model "${MODEL_PATH}/${LATEST_MODEL}"\
        --cache_path "/tmp" \
        --model_identifier "joint-mtl-seed${seed}"
} || exit 1
done

