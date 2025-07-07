import os
import json 
import logging

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
    # with open('', 'r') as f:
    #     data = f.json()
        

def generate_keys():
    '''
    _id to label, and label to _id
    this will be the master key that can be referenced across all models and all versions
    if we loose that file we are kind of screwed
    we should definitly make backups of this

    format can be json, or anything that can be parsed to a hashmap
    '''
    logging.warning('generate_keys not implemented yet')
    
    # with open('', 'w') as f:
    #     key = f.json()

def process_deckdrafterprod():
    logging.info('process_deckdrafterprod called...')
    download_images('foo')
    generate_keys()


if __name__ == '__main__':
    process_deckdrafterprod()

