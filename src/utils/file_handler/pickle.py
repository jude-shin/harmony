import os
import logging 
import pickle

from utils.product_lines import PRODUCTLINES as PLS



def load_ids(pl: PLS, prefix: str, mode: str) -> list[str]:
    '''
    Args:
        pl (PLS): TODO
        prefix (str): the name of the sub_model that is going to be used
            ex) the prefix of master_ids.pkl is 'master'
            ex) the prefix of m0_ids.pkl is 'm0'
    Returns:
        list[str]: the list of the _ids that were pickled
        
    '''
    # depickle the list of labels 
    # the index is the label, and the string inside is the _id
    data_dir = os.getenv('DATA_DIR')
    if data_dir is None:
        msg = 'DATA_DIR env var is not set...'
        logging.error(msg)
        raise KeyError(msg)

    _ids_path: str = os.path.join(data_dir, pl.value, prefix+'_ids.pkl')
    _ids: list[str] = []

    with open(_ids_path, mode) as f:
        pickle.dump(_ids, f)

    return _ids 
