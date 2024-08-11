import argparse
import datetime
from os import path

import pandas as pd
from config import SBIC_CATEGORICAL_COLUMNS
from datasets import Dataset
from icecream import ic
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    pipeline,
)
from transformers.pipelines.pt_utils import KeyDataset
from util import get_custom_timestamp_string

ic.configureOutput(prefix=get_custom_timestamp_string)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--test_data",
    type=str,
    help="Path to the test dataset in the csv format.",
    required=True,
)
parser.add_argument(
    "--base_model",
    type=str,
    help="Base model that was finetuned to retrieve the inference model. Mainly important for "
    "loading the correct tokenizer.",
    required=True,
)
parser.add_argument(
    "--groupYN_model",
    type=str,
    help="Path or name to the encoder model used for groupYN task inference.",
    required=True,
)
parser.add_argument(
    "--intentYN_model",
    type=str,
    help="Path or name to the encoder model used for intentYN task inference.",
    required=True,
)
parser.add_argument(
    "--lewdYN_model",
    type=str,
    help="Path or name to the encoder model used for lewdYN task inference.",
    required=True,
)
parser.add_argument(
    "--offensiveYN_model",
    type=str,
    help="Path or name to the encoder model used for offensiveYN task inference.",
    required=True,
)
parser.add_argument(
    "--ingroupYN_model",
    type=str,
    help="Path or name to the encoder model used for ingroupYN task inference.",
    required=True,
)
parser.add_argument(
    "--cache_path",
    type=str,
    help=(
        "Path to the directory were dataset can be cached. This should be a location with fast I/O."
    ),
    default="/tmp",
)
parser.add_argument(
    "--model_identifier",
    type=str,
    help=(
        "A unique string that is appended to the end of the output file. This makes it easier to "
        "identify them later."
    ),
    default="",
)
args = parser.parse_args()

BASE_MODEL_PATH = args.base_model
BASE_MODEL_NAME = path.basename(BASE_MODEL_PATH)
# MODEL_NAME = path.basename(args.model)
DATA_CACHE_PATH = path.join(args.cache_path)

# Hyperparameters
BATCH_SIZE = 128
MAX_SEQUENCE_LENGTH = 256


def preprocess_text(samples):
    return tokenizer(
        samples["text"], truncation=True, padding="max_length", max_length=MAX_SEQUENCE_LENGTH
    )


ic("===Loading test data")
sbic_test = pd.read_csv(args.test_data)


for task in SBIC_CATEGORICAL_COLUMNS:
    task_specific_model = args.__dict__[f"{task}_model"]
    ic("============================================================")
    ic(f"Working on task: '{task}'")
    ic(f"===Loading model '{task_specific_model}'")
    model = AutoModelForSequenceClassification.from_pretrained(
        task_specific_model,
        device_map={"": 0},  # Load the full model on GPU
        # device_map="auto",  # Load model on multiple GPUs
    )
    # We still need to load the tokenizer from the base model, instead of the finetuned one
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

    ic("===Preprocessing data")
    task_test_data = sbic_test.rename(columns={"post_preprocessed": "text", task: "label"})[
        ["text", "label"]
    ]
    test_dataset = Dataset.from_pandas(task_test_data, preserve_index=False)
    # test_dataset = get_sbic_data_as_datasets(target_label_column=task, splits=["test"])[0]
    ic("=Preparing test split")
    test_dataset = test_dataset.map(
        preprocess_text,
        batched=True,
        cache_file_name=f"{DATA_CACHE_PATH}/sbic-hf-inference-{task}-test_{args.model_identifier}.arrow",
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create sequence classification pipeline
    inference_pipe = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
    )

    ic("===Starting inference")
    test_key_dataset = KeyDataset(test_dataset, "text")

    # Generate predictions using pipeline
    predictions = [
        output for output in tqdm(inference_pipe(test_key_dataset), total=len(test_dataset))
    ]

    # Extract the label binary value from predicted string label
    prediction_column_label = f"prediction_{task}_{BASE_MODEL_NAME}-{args.model_identifier}"
    sbic_test.loc[:, prediction_column_label] = [int(pred["label"][-1]) for pred in predictions]

# Export prediction results to file
ic("===Writing results to file")
prediction_column_labels = [
    f"prediction_{t}_{BASE_MODEL_NAME}-{args.model_identifier}" for t in SBIC_CATEGORICAL_COLUMNS
]
sbic_test[["post_id", *prediction_column_labels]].to_csv(
    path.join("output", f"sbic-test_predictions-{BASE_MODEL_NAME}-{args.model_identifier}.csv"),
    sep=",",
    index=False,
)


ic("Done.")
