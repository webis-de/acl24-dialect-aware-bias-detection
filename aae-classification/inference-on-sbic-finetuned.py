from os import path

import pandas as pd
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

BASE_MODEL_PATH = "./model/deberta-v3-base"
# MODEL_PATH = "./model/deberta-v3-base-aee-classifier"
# MODEL_NAME = path.basename(MODEL_PATH)
MODEL_PATH = "./intermediate/deberta-v3-base-aee-classifier-checkpoints/checkpoint-100000"
MODEL_NAME = "deberta-v3-base-aee-classifier-cp100000"
# DATA_CACHE_PATH = path.join("intermediate", "cache")
DATA_CACHE_PATH = path.join("/tmp")

# Hyperparameters
BATCH_SIZE = 128
MAX_SEQUENCE_LENGTH = 256


def preprocess_text(samples):
    return tokenizer(
        samples["text"], truncation=True, padding="max_length", max_length=MAX_SEQUENCE_LENGTH
    )


ic(f"===Loading model '{MODEL_NAME}'")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    # device_map={"": 0},  # Load the full model on GPU
    # device_map="auto",  # Load model on multiple GPUs
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

ic("===Loading and preprocessing data")
sbic_train = pd.read_csv("../data/sbic-data/sbic-train-prep.csv")
sbic_val = pd.read_csv("../data/sbic-data/sbic-val-prep.csv")
sbic_test = pd.read_csv("../data/sbic-data/sbic-test-prep.csv")

ic("===Preprocessing data")
task_train_data = sbic_train.rename(columns={"post_preprocessed": "text"})[["text"]]
train_dataset = Dataset.from_pandas(task_train_data, preserve_index=False)
task_val_data = sbic_val.rename(columns={"post_preprocessed": "text"})[["text"]]
val_dataset = Dataset.from_pandas(task_val_data, preserve_index=False)
task_test_data = sbic_test.rename(columns={"post_preprocessed": "text"})[["text"]]
test_dataset = Dataset.from_pandas(task_test_data, preserve_index=False)

ic("=Preparing test split")
train_dataset = train_dataset.map(
    preprocess_text,
    batched=True,
    cache_file_name=f"{DATA_CACHE_PATH}/sbic-hf-inference-aae-train.arrow",
)
val_dataset = val_dataset.map(
    preprocess_text,
    batched=True,
    cache_file_name=f"{DATA_CACHE_PATH}/sbic-hf-inference-aae-val.arrow",
)
test_dataset = test_dataset.map(
    preprocess_text,
    batched=True,
    cache_file_name=f"{DATA_CACHE_PATH}/sbic-hf-inference-aae-test.arrow",
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Create sequence classification pipeline
inference_pipe = pipeline(
    task="text-classification",
    model=model,
    tokenizer=tokenizer,
    batch_size=BATCH_SIZE,
)

ic("===Starting inference")
train_key_dataset = KeyDataset(train_dataset, "text")
val_key_dataset = KeyDataset(val_dataset, "text")
test_key_dataset = KeyDataset(test_dataset, "text")

# Generate predictions using pipeline
predictions_train = [
    output for output in tqdm(inference_pipe(train_key_dataset), total=len(train_dataset))
]
predictions_val = [
    output for output in tqdm(inference_pipe(val_key_dataset), total=len(val_dataset))
]
predictions_test = [
    output for output in tqdm(inference_pipe(test_key_dataset), total=len(test_dataset))
]

# Extract the label binary value from predicted string label
sbic_train.loc[:, "aae_dialect"] = [int(pred["label"][-1]) for pred in predictions_train]
sbic_val.loc[:, "aae_dialect"] = [int(pred["label"][-1]) for pred in predictions_val]
sbic_test.loc[:, "aae_dialect"] = [int(pred["label"][-1]) for pred in predictions_test]

# Export prediction results to file
sbic_train.to_csv(
    path.join("output", f"sbic-train_aae-annotated-{MODEL_NAME}.csv"),
    sep=",",
    index=False,
)
sbic_val.to_csv(
    path.join("output", f"sbic-val_aae-annotated-{MODEL_NAME}.csv"),
    sep=",",
    index=False,
)
ic("===Writing results to file")
sbic_test.to_csv(
    path.join("output", f"sbic-test_aae-annotated-{MODEL_NAME}.csv"),
    sep=",",
    index=False,
)


ic("Done.")
