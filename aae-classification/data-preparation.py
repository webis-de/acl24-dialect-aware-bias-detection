import csv
from os import path

import pandas as pd
from config import PANDAS_NAN_STRING_VALUES
from icecream import ic
from util import get_custom_timestamp_string, preprocess_twitter_texts

ic.configureOutput(prefix=get_custom_timestamp_string)


ic("===Loading data")
# Load TwitterAAE data from file
twitter_data_full = pd.read_csv(
    path.join("..", "data", "twitter-aae", "twitteraae_all"),
    sep="\t",
    escapechar="\\",
    names=[
        "post_id",
        "timestamp",
        "user_id",
        "location",
        "census_blk_group",
        "post",
        "demograpic1_inference",
        "demograpic2_inference",
        "demograpic3_inference",
        "demograpic4_inference",
    ],
)
twitter_data_aae = pd.read_csv(
    path.join("..", "data", "twitter-aae", "twitteraae_all_aa"),
    sep="\t",
    escapechar="\\",
    names=[
        "post_id",
        "timestamp",
        "user_id",
        "location",
        "census_blk_group",
        "post",
        "demograpic1_inference",
        "demograpic2_inference",
        "demograpic3_inference",
        "demograpic4_inference",
    ],
)

ic("===Preparing data")
# Since the full dataset is a superset of the AAE dataset, we need to filter AAE samples
twitter_data_no_aae = twitter_data_full[
    ~twitter_data_full.post_id.isin(twitter_data_aae.post_id)
].copy()

# Extract columns of interest
columns_of_interest = ["post_id", "post"]
twitter_data_aae = twitter_data_aae[columns_of_interest]
twitter_data_no_aae = twitter_data_no_aae[columns_of_interest]

# Preprocess text columns
ic("==AAE data")
twitter_data_aae_prep = preprocess_twitter_texts(twitter_data_aae)
ic("==Non-AAE data")
twitter_data_no_aae_prep = preprocess_twitter_texts(twitter_data_no_aae)

# Adding aae dialect labels based on which dataframe samples are in
twitter_data_aae_prep["aae_dialect_label"] = 1
twitter_data_no_aae_prep["aae_dialect_label"] = 0

# Merge AAE and non AAE posts into a single dataframe
ic("===Merging data")
twitter_data_all_labels = twitter_data_aae_prep.merge(twitter_data_no_aae_prep, how="outer")

# Only select the columns of interest
twitter_data_all_labels = twitter_data_all_labels[
    ["post_id", "post_preprocessed", "aae_dialect_label"]
]

# Remove empty samples that might have appeared after preprocessing
ic("===Cleaning data")
twitter_data_all_labels = twitter_data_all_labels[twitter_data_all_labels.post_preprocessed != ""]
# twitter_data_all_labels = twitter_data_all_labels[
#     twitter_data_all_labels.post_preprocessed != "null"
# ]
# twitter_data_all_labels = twitter_data_all_labels[
#     twitter_data_all_labels.post_preprocessed != "nan"
# ]
# Remove values that pandas interprets as NaN values later on
twitter_data_all_labels = twitter_data_all_labels[
    ~twitter_data_all_labels.post_preprocessed.isin(PANDAS_NAN_STRING_VALUES)
]
twitter_data_all_labels.dropna(inplace=True, axis=0)

# Write final prepared data to file
ic("===Writing to files")
twitter_data_all_labels.to_csv(
    path.join("intermediate", "twitter-aae", "twitteraae-full-labeled-prep.csv"),
    sep=",",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC,
)

ic("Done.")
