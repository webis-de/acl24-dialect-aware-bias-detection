import argparse
import datetime
from os import path

import numpy as np
from config import SBIC_CATEGORICAL_COLUMNS
from datasets import load_dataset
from icecream import ic
from multitask_data_collator import MultitaskTrainer, NLPDataCollator
from multitask_model import (
    BertForSequenceClassification,
    DebertaV2ForSequenceClassification,
    RobertaForSequenceClassification,
)
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoTokenizer, TrainingArguments, set_seed
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
    help="Path or name to the base encoder model that will be finetuned.",
    required=True,
)
parser.add_argument(
    "--cache_path",
    type=str,
    help=(
        "Path to the directory were datasets can be cached. This should be a location "
        "with fast I/O"
    ),
    default="/tmp",
)
parser.add_argument(
    "--model_identifier",
    type=str,
    help=(
        "A unique string that is appended to the end of each model-specific file saved (i.e., "
        "logs, checkpoints, predictions). This makes it easier to train multiple models and "
        "evaluate them later."
    ),
    default="",
)
parser.add_argument(
    "--seed", type=int, help="Seed to be used for random initializations", default=42
)
parser.add_argument(
    "--labels",
    nargs="+",
    type=str,
    help="A whitespace-seperated list of labels to include in the learning process.",
    default=["aae_dialect", *SBIC_CATEGORICAL_COLUMNS],
)
args = parser.parse_args()

MODEL_PATH = args.base_model
MODEL_NAME = path.basename(MODEL_PATH)
DATA_CACHE_PATH = path.join(args.cache_path)
SEED = args.seed

# Hyperparameters
BATCH_SIZE = 64
TRAIN_EPOCHS = 3
MAX_SEQUENCE_LENGTH = 256
LEARNING_RATE = 1e-05

# Set seeds for reproducable results
set_seed(seed=args.seed)
# enable_full_determinism(seed=args.seed)  # significantly impacts training speed


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    return {
        "macro_F1": precision_recall_fscore_support(labels, predictions, average="macro")[2],
        "micro_F1": precision_recall_fscore_support(labels, predictions, average="micro")[2],
        "binary_F1_postive": precision_recall_fscore_support(
            labels, predictions, average="binary", pos_label=1
        )[2],
        "binary_F1_negative": precision_recall_fscore_support(
            labels, predictions, average="binary", pos_label=0
        )[2],
    }


def preprocess_text(samples):
    batch_tokenized = tokenizer(
        samples["doc"], truncation=True, padding="max_length", max_length=MAX_SEQUENCE_LENGTH
    )
    batch_tokenized["labels"] = samples["target"]
    return batch_tokenized


# Loading tokenizer
ic("===Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


# Loading data from file
ic("===Loading & Preparing data")
ic(f"==Using the following list of labels: {args.labels}")
dataset_dict = {}
task_labels_map = {}
for label in args.labels:
    ic(f"=={label}")

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


# Loading multitask model based on the model name
if "deberta" in MODEL_NAME.lower():
    ic("===Loading model as DeBERTa model.")
    model = DebertaV2ForSequenceClassification.from_pretrained(
        MODEL_PATH, task_labels_map=task_labels_map
    )
elif "roberta" in MODEL_NAME.lower():
    ic("===Loading model as RoBERTa model.")
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_PATH, task_labels_map=task_labels_map
    )
elif "bert" in MODEL_NAME.lower() and "roberta" not in MODEL_NAME.lower():
    ic("===Loading model as BERT model.")
    model = BertForSequenceClassification.from_pretrained(
        MODEL_PATH, task_labels_map=task_labels_map
    )
else:
    raise AttributeError("Unknown model type. Should either be BERT, DeBERTa or RoBERTa.")


# Extracting train datasets
train_datasets = {label: data["train"] for label, data in dataset_dict.items()}
val_dataset = {label: data["validation"] for label, data in dataset_dict.items()}
data_collator = NLPDataCollator()
if args.model_identifier == "":
    model_path_basename = f"{MODEL_NAME}-seed{SEED}"
else:
    model_path_basename = f"{MODEL_NAME}-{args.model_identifier}-seed{SEED}"

ic("===Starting model training")
ic(f"==Saving model-related files with basename {model_path_basename} in path.")
# Training multitask model
training_args = TrainingArguments(
    logging_dir=f"./logs/{model_path_basename}",
    per_device_train_batch_size=BATCH_SIZE,
    do_train=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    greater_is_better=True,
    num_train_epochs=TRAIN_EPOCHS,
    eval_steps=200,
    save_steps=200,
    logging_steps=50,
    output_dir=f"intermediate/{model_path_basename}-checkpoints",
    learning_rate=LEARNING_RATE,
)
trainer = MultitaskTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_datasets,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

ic("==Saving best model")
model_save_path = f"model/{model_path_basename}"
trainer.save_model(model_save_path)

ic("Done.")
