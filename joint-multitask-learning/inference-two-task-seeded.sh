#! /bin/bash

SEEDS=(23 42 271 314 1337)
BASE_MODEL_NAME="bert-base-uncased"
MODEL_PATH="./model/__final/multi-seed-runs"
SBIC_TASKS=("groupYN" "intentYN" "lewdYN" "offensiveYN" "ingroupYN")

for seed in "${SEEDS[@]}"; do
{
    echo -e "\e[31m==================================================\e[0m"
    echo -e "\e[31mStarting inference for seed ${seed}\e[0m"
    echo -e "\e[31m==================================================\e[0m"

    declare -A SBIC_MODELS

    for task in "${SBIC_TASKS[@]}"; do
        {
            # Find the latest trained model for the current task
            LATEST_MODEL=$(ls "${MODEL_PATH}" | grep "^${BASE_MODEL_NAME}-two-task-mtl_${task}_[0-9]\+-seed${seed}$" | sort -t '_' -k 3 -n | tail -n 1)

            SBIC_MODELS["${task}"]="${LATEST_MODEL}"
            echo -e "\e[31mChose model '${LATEST_MODEL}' for task '${task}'.\e[0m"
        } || exit 1
    done

    echo -e "\e[31m==================================================\e[0m"
    echo ""

    python inference-two-task.py \
        --data_dir "./intermediate" \
        --base_model "./model/${BASE_MODEL_NAME}" \
        --groupYN_model "${MODEL_PATH}/${SBIC_MODELS['groupYN']}"\
        --intentYN_model "${MODEL_PATH}/${SBIC_MODELS['intentYN']}"\
        --lewdYN_model "${MODEL_PATH}/${SBIC_MODELS['lewdYN']}"\
        --offensiveYN_model "${MODEL_PATH}/${SBIC_MODELS['offensiveYN']}"\
        --ingroupYN_model "${MODEL_PATH}/${SBIC_MODELS['ingroupYN']}"\
        --cache_path "/tmp" \
        --model_identifier "two-task-mtl-seed${seed}"
} || exit 1
done

