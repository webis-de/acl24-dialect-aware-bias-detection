import csv

import pandas as pd
from config import SBIC_CATEGORICAL_COLUMNS
from icecream import ic

ic("===Loading data")
sbic_train = pd.read_csv(
    "../aae-classification/output/sbic-train_aae-annotated-deberta-v3-base-aee-classifier.csv"
)
sbic_val = pd.read_csv(
    "../aae-classification/output/sbic-val_aae-annotated-deberta-v3-base-aee-classifier.csv"
)
sbic_test = pd.read_csv(
    "../aae-classification/output/sbic-test_aae-annotated-deberta-v3-base-aee-classifier.csv"
)

# required output: for each label, a tsv file with "id", "doc", "target" fields
ic("===Separating data per label")
for label in ["aae_dialect", *SBIC_CATEGORICAL_COLUMNS]:
    rename_dict = {"post_id": "id", "post_preprocessed": "doc", label: "target"}

    # Select column subset and rename columns
    mtl_train_label_df = (
        sbic_train[["post_id", "post_preprocessed", label]]
        .copy()
        .rename(columns=rename_dict)
    )
    mtl_val_label_df = (
        sbic_val[["post_id", "post_preprocessed", label]]
        .copy()
        .rename(columns=rename_dict)
    )
    mtl_test_label_df = (
        sbic_test[["post_id", "post_preprocessed", label]]
        .copy()
        .rename(columns=rename_dict)
    )

    with open(f"intermediate/sbic-train_mtl-{label}.tsv", "w") as f:
        mtl_train_label_df.to_csv(
            f, sep="\t", index=False, quoting=csv.QUOTE_NONNUMERIC
        )

    with open(f"intermediate/sbic-val_mtl-{label}.tsv", "w") as f:
        mtl_val_label_df.to_csv(f, sep="\t", index=False, quoting=csv.QUOTE_NONNUMERIC)

    with open(f"intermediate/sbic-test_mtl-{label}.tsv", "w") as f:
        mtl_test_label_df.to_csv(f, sep="\t", index=False, quoting=csv.QUOTE_NONNUMERIC)


ic("Done.")
