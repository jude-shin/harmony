from time import localtime, strftime, time

def get_current_time() -> str:
    '''
    Returns the current time as a nice string.

    Args:
    Returns:
        str: the current time as a string 
    '''


    '''
    Convert a given label to the deckdrafterprod _id based on the master_labels.toml

    Args:
        label (str): what the tensorflow model will spit out
        pl (PRODUCTLINES): The product_line we are working with.
    Returns:
        str: _id that is associated with that label
    '''

    return strftime('%B %d, %I:%M%p', localtime())


def get_elapsed_time(start_time: float) -> str:
    '''
    Using seconds since epoch, determine how much time has passed since the provided float. 
    At the beginning of the operation you want to time, import time.time(), and that should return a float that you can pass into this function
    
    Args:
        float: the starting time (what time we are referencing)
    Returns:
        str: hours:minutes:seconds
    '''
    elapsed_seconds = time() - start_time
    h = int(elapsed_seconds / 3600)
    m = int((elapsed_seconds - h * 3600) / 60)
    s = int((elapsed_seconds - m * 60) - h * 3600)
    return f'{h}hr {m}m {s}s'

