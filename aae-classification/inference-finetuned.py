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
    device_map={"": 0},  # Load the full model on GPU
    # device_map="auto",  # Load model on multiple GPUs
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

ic("===Loading and preprocessing data")
# data_cache_file = path.join(DATA_CACHE_PATH, "aae-hf-test-inference.arrow")
# if not path.exists(data_cache_file):
# ic("==No cache file found")
# test_dataset = get_twitteraae_data_as_datasets(splits=["test"])["test"]
twitteraae_test = pd.read_csv("intermediate/twitter-aae/twitteraae-test-labeled-prep.csv")
twitteraae_test = twitteraae_test.rename(
    columns={"post_preprocessed": "text", "aae_dialect_label": "label"}
)
test_dataset = Dataset.from_pandas(twitteraae_test[["text", "label"]], preserve_index=False)
ic("=Preparing test split")
test_dataset = test_dataset.map(
    preprocess_text, batched=True, cache_file_name=f"{DATA_CACHE_PATH}/aae-hf-test-inference.arrow"
)

#     ic("==Caching data")
#     with open(data_cache_file, "wb") as f:
#         pickle.dump((test_dataset, twitteraae_test), f)
# else:
#     ic("==Loading from cache file")
#     with open(data_cache_file, "rb") as f:
#         test_dataset, twitteraae_test = pickle.load(f)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Create sequence classification pipeline
inference_pipe = pipeline(
    task="text-classification",
    model=model,
    tokenizer=tokenizer,
    batch_size=BATCH_SIZE,
)


ic("===Starting inference")
test_key_dataset = KeyDataset(test_dataset, "text")

# Generate predictions using pipeline
predictions = [output for output in tqdm(inference_pipe(test_key_dataset), total=len(test_dataset))]

# Extract the label binary value from predicted string label
prediction_column_label = f"prediction_{MODEL_NAME}"
twitteraae_test.loc[:, prediction_column_label] = [int(pred["label"][-1]) for pred in predictions]

# Export prediction results to file
ic("===Writing results to file")
twitteraae_test[["post_id", prediction_column_label]].to_csv(
    path.join("output", f"twitteraae-test_predictions-{MODEL_NAME}.csv"),
    sep=",",
    index=False,
)

ic("Done.")
