import argparse
from os import path

import numpy as np
import torch
from config import SBIC_CATEGORICAL_COLUMNS
from icecream import ic
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from util import get_custom_timestamp_string, get_sbic_data_as_datasets

ic.configureOutput(prefix=get_custom_timestamp_string)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_data",
    type=str,
    help="Path to the train dataset in the csv format.",
    required=True,
)
parser.add_argument(
    "--val_data",
    type=str,
    help="Path to the validation dataset in the csv format.",
    required=True,
)
parser.add_argument(
    "--test_data",
    type=str,
    help="Path to the test dataset in the csv format.",
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
args = parser.parse_args()


MODEL_PATH = args.base_model
MODEL_NAME = path.basename(MODEL_PATH)
DATA_CACHE_PATH = path.join(args.cache_path)
SEED = args.seed

# Hyperparameters
BATCH_SIZE = 64
TRAIN_EPOCHS = 3
MAX_SEQUENCE_LENGTH = 256

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
    return tokenizer(
        samples["text"], truncation=True, padding="max_length", max_length=MAX_SEQUENCE_LENGTH
    )


for task in SBIC_CATEGORICAL_COLUMNS:
    ic("============================================================")
    ic(f"Working on task: '{task}'")
    ic(f"===Loading model '{MODEL_NAME}'")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        # device_map={"": 0},  # Load the full model on GPU
        # device_map="auto",  # Load model on multiple GPUs
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    ic("===Loading and preprocessing data")
    train_dataset, val_dataset, test_dataset = get_sbic_data_as_datasets(
        target_label_column=task,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        test_data_path=args.test_data,
    )
    ic("=Preparing train split")
    train_dataset = train_dataset.map(
        preprocess_text,
        batched=True,
        cache_file_name=f"{DATA_CACHE_PATH}/sbic-hf-{task}-train_{args.model_identifier}.arrow",
        num_proc=12,
    )
    ic("=Preparing val split")
    val_dataset = val_dataset.map(
        preprocess_text,
        batched=True,
        cache_file_name=f"{DATA_CACHE_PATH}/sbic-hf-{task}-val_{args.model_identifier}.arrow",
        num_proc=12,
    )
    ic("=Preparing test split")
    test_dataset = test_dataset.map(
        preprocess_text,
        batched=True,
        cache_file_name=f"{DATA_CACHE_PATH}/sbic-hf-{task}-test_{args.model_identifier}.arrow",
        num_proc=12,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    if args.model_identifier == "":
        model_path_basename = f"{MODEL_NAME}-{task}-seed{SEED}"
    else:
        model_path_basename = f"{MODEL_NAME}-{task}-{args.model_identifier}-seed{SEED}"

    ic("===Starting model training")
    ic(f"==Saving model-related files with basename {model_path_basename} in path.")
    training_args = TrainingArguments(
        logging_dir=f"./logs/{model_path_basename}",
        per_device_train_batch_size=BATCH_SIZE,
        do_eval=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        greater_is_better=True,
        num_train_epochs=TRAIN_EPOCHS,
        eval_steps=200,
        save_steps=500,
        logging_steps=10,
        output_dir=f"intermediate/{model_path_basename}-checkpoints",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=val_dataset,
        train_dataset=train_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    ic("==Saving best model")
    model_save_path = f"model/{model_path_basename}"
    trainer.save_model(model_save_path)

    # Cleaning GPU memory for next run
    del model, tokenizer, train_dataset, val_dataset, test_dataset, trainer, training_args
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


ic("Done.")
