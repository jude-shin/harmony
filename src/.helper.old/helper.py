import os
from time import localtime, strftime, time
import hashlib


# =======================================
# TIME
def get_current_time():
    """Help: Returns the current time as a nice string."""
    return strftime("%B %d, %I:%M%p", localtime())


def get_elapsed_time(start_time):
    """Using seconds since epoch, determine how much time has passed since the provided float. Returns string
    with hours:minutes:seconds"""
    elapsed_seconds = time() - start_time
    h = int(elapsed_seconds / 3600)
    m = int((elapsed_seconds - h * 3600) / 60)
    s = int((elapsed_seconds - m * 60) - h * 3600)
    return f"{h}hr {m}m {s}s"


# =======================================
# FILES
def generate_unique_filename(directory, base_name, extension):
    counter = 0
    while True:
        filename = f"{base_name}({counter}).{extension}"
        if not os.path.exists(os.path.join(directory, filename)):
            return filename
        counter += 1


# =======================================
# LABELS
def alphanumeric_to_int(alp_num_str: str):
    try:
        result = int(hashlib.sha256(
            alp_num_str.encode()).hexdigest(), 16) % 10**8
    except ValueError:
        raise ValueError("Invalid alphanumeric string provided.")

    return result
