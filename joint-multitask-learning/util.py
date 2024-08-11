import time


def get_custom_timestamp_string():
    return "%s |> " % time.strftime("%Y-%m-%d--%H:%M:%S")
