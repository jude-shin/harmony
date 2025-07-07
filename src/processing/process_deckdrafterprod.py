import os
import json 
import logging

from utils.product_lines import PRODUCTLINES as PLS

# TODO: renme this file to preprocessing or something
# TODO: function that adds images that already have an _id
    # checks to see that the key is already present, (skips the image if the key is unknown)
    # downloads the image with the _id as the name 
    # if there is already an image with that name, add a (n) to the end


# TODO: we are going to add all the photos to the queue side
# all this data is temporary because it will be turned into a tf.data.Dataset
# but maybe we should still save them to reduce the overhead of retraining when the time comes to retrain the whole thing?


def download_images(deckdrafterprod_path):
    '''
    NOTE: this is used only if there are new cards with new _ids added to the deckdrafterprod
    IT IS NOT used for when we want to download images of cards that we know the _id of already

    for each element in a json that is given ()
    check to see if we have the key already (this means that there is already a downloaded image)

    if the key is not registerd, then download an image
    add it to a processed_deckdrafterprod.json
    each name of the card should be the _id
    '''
    logging.warning('download_images not implemented yet')


        
        

def generate_keys(pl: PLS):
    '''
    label_to_id 
        NOTE: if we need the id_to_label, you can load the json to a dict, and then zip the values and the keys
        inverted_dict = dict(zip(original_dict.values(), original_dict.keys()))
    this will be the master key that can be referenced across all models and all versions
    if we loose that file we are kind of screwed
    we should definitly make backups of this

    format can be json, or anything that can be parsed to a hashmap
    '''
    # logging.warning('generate_keys not implemented yet')
    deckdrafterprod_path = os.path.join(os.getenv('DATADIR'), pl.value, 'deckdrafterprod.json')
    with open(deckdrafterprod_path, 'r') as f:
        deckdrafterprod = f.json()

    label_to_id = {}
    
    label = -1
    for card in data:
        _id = card['_id']
        label += 1
        label_to_id[str(label)] = str(_id)
    
    label_to_id_path = os.path.join(os.getenv('DATADIR'), pl.value, 'label_to_id.json')
    with open(label_to_id_path, 'w+') as f:
        json.dump(data, f, indent=4)


def process_deckdrafterprod():
    logging.info('process_deckdrafterprod called...')
    download_images('foo')
    generate_keys(PLS.LORCANA) # TODO: hardcoded for now

if __name__ == '__main__':
    process_deckdrafterprod()

