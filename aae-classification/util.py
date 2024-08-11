import time

import pandas as pd
from config import RANDOM_STATE, TEST_SIZE, VAL_SIZE
from datasets import ClassLabel, Dataset, Value
from datasets.features.features import Features
from sklearn.model_selection import train_test_split


def get_custom_timestamp_string():
    return "%s |> " % time.strftime("%Y-%m-%d--%H:%M:%S")


def get_parallel_data_as_datasets() -> tuple:
    # Load data from file
    with open("data/sae-aave-pairs/sae_samples.txt", "r") as f:
        sae_texts = [{"text": s, "label": "class_0"} for s in f.readlines()]
    with open("data/sae-aave-pairs/aave_samples.txt", "r") as f:
        aae_texts = [{"text": s, "label": "class_1"} for s in f.readlines()]
    texts_with_labels = [*sae_texts, *aae_texts]

    # Generate random splits
    X = [sample["text"] for sample in texts_with_labels]
    y = [sample["label"] for sample in texts_with_labels]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=RANDOM_STATE, shuffle=True, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, shuffle=True, stratify=y_train
    )
    texts_train = [{"text": X_train[i], "label": y_train[i]} for i in range(len(X_train))]
    texts_val = [{"text": X_val[i], "label": y_val[i]} for i in range(len(X_val))]
    texts_test = [{"text": X_test[i], "label": y_test[i]} for i in range(len(X_test))]

    # Transform into HF dataset
    features = Features(
        {"text": Value("string"), "label": ClassLabel(num_classes=2, names=["class_0", "class_1"])}
    )

    train_dataset = Dataset.from_list(texts_train, features=features)
    val_dataset = Dataset.from_list(texts_val, features=features)
    test_dataset = Dataset.from_list(texts_test, features=features)
    # data = dataset.train_test_split(test_size=0.2, shuffle=True)

    # data = Dataset.from_list(texts_with_labels, features=features).train_test_split(
    #     test_size=0.2, shuffle=True, stratify_by_column="label"
    # )

    return (train_dataset, val_dataset, test_dataset)
    # return data


def get_twitteraae_data_as_datasets(splits: list = ["train", "val", "test"]) -> list:
    data_splits = []

    if "train" in splits:
        twitteraae_train = pd.read_csv("intermediate/twitter-aae/twitteraae-train-labeled-prep.csv")
        twitteraae_train = twitteraae_train.rename(
            columns={"post_preprocessed": "text", "aae_dialect_label": "label"}
        )[["text", "label"]]
        train_dataset = Dataset.from_pandas(twitteraae_train, preserve_index=False)

        data_splits.append(train_dataset)

    if "val" in splits:
        twitteraae_val = pd.read_csv("intermediate/twitter-aae/twitteraae-val-labeled-prep.csv")
        twitteraae_val = twitteraae_val.rename(
            columns={"post_preprocessed": "text", "aae_dialect_label": "label"}
        )[["text", "label"]]
        val_dataset = Dataset.from_pandas(twitteraae_val, preserve_index=False)

        data_splits.append(val_dataset)

    if "test" in splits:
        twitteraae_test = pd.read_csv("intermediate/twitter-aae/twitteraae-test-labeled-prep.csv")
        twitteraae_test = twitteraae_test.rename(
            columns={"post_preprocessed": "text", "aae_dialect_label": "label"}
        )[["text", "label"]]
        test_dataset = Dataset.from_pandas(twitteraae_test, preserve_index=False)

        data_splits.append(test_dataset)

    return data_splits


# def get_twitteraae_data_as_datasets() -> tuple:
#     # Load data from file; we don't need the na-filter, as data is already cleaned
#     # twitteraae_full = pd.read_csv("intermediate/twitter-aae/twitteraae-full-labeled-prep.csv")
#     twitteraae_train = pd.read_csv("intermediate/twitter-aae/twitteraae-train-labeled-prep.csv")
#     twitteraae_val = pd.read_csv("intermediate/twitter-aae/twitteraae-val-labeled-prep.csv")
#     twitteraae_test = pd.read_csv("intermediate/twitter-aae/twitteraae-test-labeled-prep.csv")

#     # Rename and sub-sample columns
#     twitteraae_train = twitteraae_train.rename(
#         columns={"post_preprocessed": "text", "aae_dialect_label": "label"}
#     )[["text", "label"]]
#     twitteraae_val = twitteraae_val.rename(
#         columns={"post_preprocessed": "text", "aae_dialect_label": "label"}
#     )[["text", "label"]]
#     twitteraae_test = twitteraae_test.rename(
#         columns={"post_preprocessed": "text", "aae_dialect_label": "label"}
#     )[["text", "label"]]
#     # X = twitteraae_full["text"]
#     # y = twitteraae_full["label"]

#     # # Generate pseudo-random splits
#     # X_train, X_test, y_train, y_test = train_test_split(
#     #     X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True, stratify=y
#     # )
#     # X_train, X_val, y_train, y_val = train_test_split(
#     #     X_train,
#     #     y_train,
#     #     test_size=VAL_SIZE,
#     #     random_state=RANDOM_STATE,
#     #     shuffle=True,
#     #     stratify=y_train,
#     # )

#     # texts_train = [{"text": X_train[i], "label": y_train[i]} for i in range(len(X_train))]
#     # texts_val = [{"text": X_val[i], "label": y_val[i]} for i in range(len(X_val))]
#     # texts_test = [{"text": X_test[i], "label": y_test[i]} for i in range(len(X_test))]

#     # Transform into HF dataset
#     # features = Features({"text": Value("string"), "label": ClassLabel(num_classes=2, names=[0, 1])})

#     # train_dataset = Dataset.from_list(texts_train, features=features)
#     # train_dataset = Dataset.from_pandas(pd.concat([X_train, y_train], axis=1), preserve_index=False)
#     train_dataset = Dataset.from_pandas(twitteraae_train, preserve_index=False)
#     # val_dataset = Dataset.from_list(texts_val, features=features)
#     # val_dataset = Dataset.from_pandas(pd.concat([X_val, y_val], axis=1), preserve_index=False)
#     val_dataset = Dataset.from_pandas(twitteraae_val, preserve_index=False)
#     # test_dataset = Dataset.from_list(texts_test, features=features)
#     # test_dataset = Dataset.from_pandas(pd.concat([X_test, y_test], axis=1), preserve_index=False)
#     test_dataset = Dataset.from_pandas(twitteraae_test, preserve_index=False)

#     return (train_dataset, val_dataset, test_dataset)


def preprocess_twitter_texts(
    dataframe: pd.DataFrame,
    unprocessed_text_column: str = "post",
    target_text_colum: str = "post_preprocessed",
) -> pd.DataFrame:
    """Clean the texts of the given dataframe in the given column.

    This creates an internal copy of the dataframe and returns it. While this might require more
    memory, it should also be faster, as it works with pandas built-in string functions.

    Return a copy of the provided dataframe with an additional column of clean text.
    """
    # Create an internal copy of the provided dataframe
    internal_df = dataframe.copy()

    # Lowercase all strings
    internal_df[target_text_colum] = internal_df[unprocessed_text_column].str.lower()
    # Remove twitter URLs(containing t.co)
    # internal_df[target_text_colum] = internal_df[target_text_colum].str.replace(
    #     "https?:\/\/t.co\/[A-Za-z0-9]+", "", regex=True
    # )
    # Remove texts like : <some text>; might interfere with special tokens
    internal_df[target_text_colum] = internal_df[target_text_colum].str.replace(
        "<.*?>", "", regex=True
    )
    # Remove #hashtags
    # internal_df[target_text_colum] = internal_df[target_text_colum].str.replace(
    #     "#\w*", "", regex=True
    # )
    # Remove @mentions
    internal_df[target_text_colum] = internal_df[target_text_colum].str.replace(
        "@[^\s @]+", "", regex=True
    )
    # Remove all special characters, emojis etc. Everything except alphanumeric and white spaces
    # internal_df[target_text_colum] = internal_df[target_text_colum].str.replace(
    #     "[^\w \s]", "", regex=True
    # )
    # Remove underscores
    # internal_df[target_text_colum] = internal_df[target_text_colum].str.replace("_", "", regex=True)
    # Remove asian language characters, such as Chinese or Japanses
    # (we are focused on English for now)
    internal_df[target_text_colum] = internal_df[target_text_colum].str.replace(
        r"[\u4e00-\u9fff]+", "", regex=True
    )
    # Replace single and continuous multiple whitespaces with single space
    internal_df[target_text_colum] = internal_df[target_text_colum].str.replace(
        "\s\s*", " ", regex=True
    )
    # Remove the word "rt", indicating retweets
    internal_df[target_text_colum] = internal_df[target_text_colum].str.replace(
        " rt ", "", regex=True
    )
    # Remove URLS with http/s
    internal_df[target_text_colum] = internal_df[target_text_colum].str.replace(
        "http[\w\d]*", "", regex=True
    )
    # Trims whitespaces on the left and right.
    internal_df[target_text_colum] = internal_df[target_text_colum].str.strip()

    return internal_df
