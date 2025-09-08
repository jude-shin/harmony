import os
import logging
import re

def generate_unique_version(base_dir: str, base_name: str, suffix: str):
    base_name = re.sub(r'\([^)]*\)', '', base_name) 
    base_path = os.path.join(base_dir, base_name+suffix)

    if not os.path.exists(base_path):
        logging.warning('Base Path (%s) does not exist', base_path)

    new_path = base_path
    version: int = 0
    while os.path.exists(new_path):
        new_path = base_path + '(' + str(version) + ')' + suffix
    
    return new_path
