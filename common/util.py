# ##################################################################################################
#
# File that contains some utility functions that are used across the codebase.
#
# ##################################################################################################

import pandas as pd


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

    internal_df[target_text_colum] = internal_df[unprocessed_text_column].str.lower()

    # Clean outer whitespaces
    internal_df[target_text_colum] = internal_df[target_text_colum].str.strip()

    # Remove twitter URLs(containing t.co)
    internal_df[target_text_colum] = internal_df[target_text_colum].str.replace(
        "https?:\/\/t.co\/[A-Za-z0-9]+", "", regex=True
    )

    # Remove texts like : <some text>
    internal_df[target_text_colum] = internal_df[target_text_colum].str.replace(
        "<.*?>", "", regex=True
    )

    # Remove #hashtags
    internal_df[target_text_colum] = internal_df[target_text_colum].str.replace(
        "#\w*", "", regex=True
    )

    # Remove @mentions
    internal_df[target_text_colum] = internal_df[target_text_colum].str.replace(
        "@[^\s @]+", "", regex=True
    )

    # Remove all special characters, emojis etc. Everything except alphanumeric and white spaces
    internal_df[target_text_colum] = internal_df[target_text_colum].str.replace(
        "[^\w \s]", "", regex=True
    )

    # Remove underscores
    internal_df[target_text_colum] = internal_df[target_text_colum].str.replace("_", "", regex=True)

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
