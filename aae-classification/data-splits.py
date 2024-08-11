import csv
from os import path

import pandas as pd
from config import RANDOM_STATE, TEST_SIZE, VAL_SIZE
from icecream import ic
from sklearn.model_selection import train_test_split
from util import get_custom_timestamp_string

ic.configureOutput(prefix=get_custom_timestamp_string)


ic("===Loading data")
twitteraae_full = pd.read_csv("intermediate/twitter-aae/twitteraae-full-labeled-prep.csv")


# Split data intro pre-defined splits
X_train, X_test, y_train, y_test = train_test_split(
    twitteraae_full,
    twitteraae_full["aae_dialect_label"],
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True,
    stratify=twitteraae_full["aae_dialect_label"],
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=VAL_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True,
    stratify=y_train,
)


# Write final prepared data to file
ic("===Writing to files")
X_train.to_csv(
    path.join("intermediate", "twitter-aae", "twitteraae-train-labeled-prep.csv"),
    sep=",",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)
X_val.to_csv(
    path.join("intermediate", "twitter-aae", "twitteraae-val-labeled-prep.csv"),
    sep=",",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)
X_test.to_csv(
    path.join("intermediate", "twitter-aae", "twitteraae-test-labeled-prep.csv"),
    sep=",",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)

ic("Done.")
