import argparse
from os import path

import pandas as pd
import torch
from config import SBIC_CATEGORICAL_COLUMNS
from datasets import load_dataset
from icecream import ic
from multitask_data_collator import NLPDataCollator
from multitask_model import (
    BertForSequenceClassification,
    DebertaV2ForSequenceClassification,
    RobertaForSequenceClassification,
)
from tqdm import tqdm
from transformers import AutoTokenizer
from util import get_custom_timestamp_string

ic.configureOutput(prefix=get_custom_timestamp_string)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    type=str,
    help="Path to the directory containing the prepared train, val and test split data files.",
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
    "--model",
    type=str,
    help="Path or name to the encoder model used for inference.",
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
MODEL_NAME = path.basename(args.model)
DATA_CACHE_PATH = path.join(args.cache_path)

# Hyperparameters
BATCH_SIZE = 64
MAX_SEQUENCE_LENGTH = 256


def preprocess_text(samples):
    batch_tokenized = tokenizer(
        samples["doc"], truncation=True, padding="max_length", max_length=MAX_SEQUENCE_LENGTH
    )
    batch_tokenized["labels"] = samples["target"]
    return batch_tokenized


# Loading tokenizer
ic("===Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)


# Load the SBIC test dataframe
sbic_test = pd.read_csv("../data/sbic-data/sbic-test-prep.csv")
sbic_test = sbic_test[["post_id"]]

# Loading data from file
ic("===Loading & Preparing data")
dataset_dict = {}
task_labels_map = {}
for label in SBIC_CATEGORICAL_COLUMNS:
    ic(f"===Label: {label}")

    # Load each dataset from file
    label_dataset = load_dataset(
        "multitask_dataloader.py",
        data_files={
            "train": f"{args.data_dir}/sbic-train_mtl-{label}.tsv",
            "validation": f"{args.data_dir}/sbic-val_mtl-{label}.tsv",
            "test": f"{args.data_dir}/sbic-test_mtl-{label}.tsv",
        },
        cache_dir=DATA_CACHE_PATH,
    ).with_format("torch")

    # Make each dataset a map object on the preprocess function
    dataset_dict[label] = {}
    for split, split_data in label_dataset.items():
        label_split_data = split_data.map(
            preprocess_text,
            batched=True,
            cache_file_name=f"{DATA_CACHE_PATH}/sbic-{label}-{split}_{args.model_identifier}.arrow",
        )
        label_split_data.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )
        dataset_dict[label][split] = label_split_data

    # Also add the current label to task-label-map (which task has how many possible labels)
    task_labels_map[label] = 2
    # Loading model components
    if "deberta" in MODEL_NAME.lower():
        ic("===Loading model as DeBERTa model.")
        model = DebertaV2ForSequenceClassification.from_pretrained(
            args.model, task_labels_map=task_labels_map
        )
    elif "roberta" in MODEL_NAME.lower():
        ic("===Loading model as RoBERTa model.")
        model = RobertaForSequenceClassification.from_pretrained(
            args.model, task_labels_map=task_labels_map
        )
    elif "bert" in MODEL_NAME.lower() and "roberta" not in MODEL_NAME.lower():
        ic("===Loading model as BERT model.")
        model = BertForSequenceClassification.from_pretrained(
            args.model, task_labels_map=task_labels_map
        )
    else:
        raise AttributeError(
            f"Unknown model type. Should either be BERT, DeBERTa or RoBERTa. Got {MODEL_NAME.lower()}"
        )
    data_collator = NLPDataCollator()

    ic("==Starting model inference")
    test_data = dataset_dict[label]["test"]
    test_predictions = []

    for batch in tqdm(
        test_data.iter(batch_size=BATCH_SIZE), total=int(len(test_data) / BATCH_SIZE)
    ):
        try:
            logits = model(
                batch["input_ids"], task_name=label, attention_mask=batch["attention_mask"]
            )[0]
        except IndexError as e:
            ic(f"Got error {e}. Trying to access logits as dict.")
            logits = model(
                batch["input_ids"], task_name=label, attention_mask=batch["attention_mask"]
            )["logits"]

        batch_predictions = torch.argmax(
            torch.FloatTensor(torch.softmax(logits, dim=1).detach().cpu().tolist()), dim=1
        )

        test_predictions.extend(batch_predictions.tolist())
    sbic_test[f"prediction_{label}_{BASE_MODEL_NAME}-{args.model_identifier}"] = test_predictions

sbic_test.to_csv(
    path.join("output", f"sbic-test_predictions-{BASE_MODEL_NAME}-{args.model_identifier}.csv"),
    sep=",",
    index=False,
)


ic("Done.")
