import argparse
import datetime
from os import path

import evaluate
import numpy as np
import torch
from icecream import ic
from sklearn.metrics import precision_recall_fscore_support
from torch import autocast, nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    # enable_full_determinism,
    set_seed,
)
from util import get_custom_timestamp_string, get_twitteraae_data_as_datasets

ic.configureOutput(prefix=get_custom_timestamp_string)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--seed", type=int, help="Seed to be used for random initializations", default=42
)
args = parser.parse_args()

MODEL_PATH = "./model/deberta-v3-base"
MODEL_NAME = path.basename(MODEL_PATH)
METRIC = evaluate.load("../hf-evaluate/metrics/f1/f1.py")
# DATA_CACHE_PATH = path.join("intermediate", "cache")
DATA_CACHE_PATH = path.join("/tmp")
SEED = args.seed

# Hyperparameters
BATCH_SIZE = 270
TRAIN_EPOCHS = 3
MAX_SEQUENCE_LENGTH = 256

# Set seeds for reproducable results
set_seed(seed=args.seed)
# enable_full_determinism(seed=args.seed)


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


ic(f"===Loading model '{MODEL_NAME}'")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    # device_map={"": 0},  # Load the full model on GPU
    # device_map="auto",  # Load model on multiple GPUs
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


ic("===Loading and preprocessing data")
train_dataset, val_dataset, test_dataset = get_twitteraae_data_as_datasets()
ic("=Preparing train split")
train_dataset = train_dataset.map(
    preprocess_text,
    batched=True,
    # load_from_cache_file=False,
    cache_file_name=f"{DATA_CACHE_PATH}/aae-hf-train.arrow",
    num_proc=12,
)
ic("=Preparing val split")
val_dataset = val_dataset.map(
    preprocess_text,
    batched=True,
    # load_from_cache_file=False,
    cache_file_name=f"{DATA_CACHE_PATH}/aae-hf-val.arrow",
    num_proc=12,
)
ic("=Preparing test split")
test_dataset = test_dataset.map(
    preprocess_text,
    batched=True,
    # load_from_cache_file=False,
    cache_file_name=f"{DATA_CACHE_PATH}/aae-hf-test.arrow",
    num_proc=12,
)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# Define a custom trainer to be able to define label weights for the loss
ic("===Calculating loss weights")
label_counts = np.unique(train_dataset["label"], return_counts=True)
label_distribution = [
    label_counts[1][0] / len(train_dataset["label"]),
    label_counts[1][1] / len(train_dataset["label"]),
]
label_weights = [float(label_distribution[1]), float(label_distribution[0])]

ic(f"==Label counts: {label_counts}")
ic(f"==Label distribution: {label_distribution}")
ic(f"==Label weights (reversed distribution): {label_weights}")


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        with autocast(device_type="cuda", dtype=torch.float16):
            labels = inputs.pop("labels")

            # forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")

            # compute custom loss (suppose one has 3 labels with different weights)
            loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(label_weights, device="cuda"))
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss


ic("===Starting model training")
training_args = TrainingArguments(
    logging_dir=f"./logs/{MODEL_NAME}-aae-classifier-seed{SEED}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
    per_device_train_batch_size=BATCH_SIZE,
    do_eval=True,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="binary_F1_postive",
    greater_is_better=True,
    num_train_epochs=TRAIN_EPOCHS,
    eval_steps=20000,
    save_steps=20000,
    logging_steps=100,
    output_dir=f"intermediate/{MODEL_NAME}-aee-classifier-seed{SEED}-checkpoints",
    fp16=True,
)
trainer = CustomTrainer(
    model=model,
    args=training_args,
    eval_dataset=val_dataset,
    train_dataset=train_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

ic("==Saving best model")
model_save_path = f"model/{MODEL_NAME}-aee-classifier-seed{SEED}"
trainer.save_model(model_save_path)

ic("Done.")
