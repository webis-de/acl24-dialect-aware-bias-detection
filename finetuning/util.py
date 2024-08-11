import time

import pandas as pd
from datasets import Dataset


def get_custom_timestamp_string():
    return "%s |> " % time.strftime("%Y-%m-%d--%H:%M:%S")


def get_sbic_data_as_datasets(
    target_label_column: str,
    splits: list = ["train", "val", "test"],
    train_data_path: str = "../data/sbic-data/sbic-train-prep.csv",
    val_data_path: str = "../data/sbic-data/sbic-val-prep.csv",
    test_data_path: str = "../data/sbic-data/sbic-test-prep.csv",
) -> list:
    data_splits = []

    if "train" in splits:
        sbic_train = pd.read_csv(train_data_path)
        sbic_train = sbic_train.rename(
            columns={"post_preprocessed": "text", target_label_column: "label"}
        )[["text", "label"]]
        train_dataset = Dataset.from_pandas(sbic_train, preserve_index=False)

        data_splits.append(train_dataset)

    if "val" in splits:
        sbic_val = pd.read_csv(val_data_path)
        sbic_val = sbic_val.rename(
            columns={"post_preprocessed": "text", target_label_column: "label"}
        )[["text", "label"]]
        val_dataset = Dataset.from_pandas(sbic_val, preserve_index=False)

        data_splits.append(val_dataset)

    if "test" in splits:
        sbic_test = pd.read_csv(test_data_path)
        sbic_test = sbic_test.rename(
            columns={"post_preprocessed": "text", target_label_column: "label"}
        )[["text", "label"]]
        test_dataset = Dataset.from_pandas(sbic_test, preserve_index=False)

        data_splits.append(test_dataset)

    return data_splits
