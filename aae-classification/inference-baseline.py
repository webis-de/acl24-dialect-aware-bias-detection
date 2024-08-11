import sys
from os import chdir, path

import numpy as np
import pandas as pd
from icecream import ic
from tqdm import tqdm
from util import get_custom_timestamp_string

chdir("twitteraae/code")
import twitteraae.code.predict as predict

ic.configureOutput(prefix=get_custom_timestamp_string)
tqdm.pandas()

script_path = path.dirname(path.realpath(sys.argv[0]))
PATH_PREFIX = f"{script_path}/../../"


def predict_dialect(text: str, model):
    """Return 1, if AAE is most likely class; 0 otherwise"""
    probabilities = model.predict(text.split())

    # AAE probability is located at the first position of the returned list of probabilities
    return 1 if np.argmax(probabilities) == 0 else 0


# Load TwitterAAE prediction model
ic("===Loading model")
predict.load_model()

ic("===Loading data")
twitteraae_test = pd.read_csv(
    f"{PATH_PREFIX}/intermediate/twitter-aae/twitteraae-test-labeled-prep.csv"
)


ic("===Running inference")
twitteraae_test["prediction_twitteraae_baseline"] = twitteraae_test[
    "post_preprocessed"
].progress_apply(predict_dialect, model=predict)


# Write final prepared data to file
ic("===Writing to file")
twitteraae_test[["post_id", "prediction_twitteraae_baseline"]].to_csv(
    path.join(PATH_PREFIX, "output", "twitteraae-test_predictions-baseline.csv"),
    sep=",",
    index=False,
)

ic("Done.")
