RANDOM_STATE = 42
VAL_SIZE = 0.2
TEST_SIZE = 0.2

# String values that pandas interprets as NaN, instead of strings, i.e. when reading from a .csv
# file; see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
PANDAS_NAN_STRING_VALUES = [
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    "NA",
    "NULL",
    "NaN",
    "n/a",
    "nan",
    "null",
]
