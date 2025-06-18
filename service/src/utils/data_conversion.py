import json
import logging
from config.constants import GAMES

# label: what the tensorflow model will spit out
# _id: tcg/deckdrafterprod unique _id
#   allows us to get more information about the card


# given a label (string), and a Game type
# return the _id (string) that is associated with that label
def label_to_id(label : str, g : GAMES) -> str:
    if not isinstance(label, str):
        raise TypeError("label_to_id expected a [str] but got [" + type(label) + "]")
    if not isinstance(g, GAMES):
        raise TypeError("reformat_json  expected a [GAMES] enum but got [" + type(g) + "]")

    json = label_to_json(label)

    # extract only the _id from the json object
    return ""

# given an _id (string), and a Game type
# return the label (string) that is associated with that _id
def id_to_label(_id : str, g : GAMES) -> str:
    if not isinstance(_id, str):
        raise TypeError("id_to_label expected a [str] but got [" + type(_id) + "]")
    if not isinstance(g, GAMES):
        raise TypeError("reformat_json  expected a [GAMES] enum but got [" + type(g) + "]")

    # look up the variable
    return ""

# given a label (string), and a Game type
# return a json object with certain fields
def label_to_json(label : str, g : GAMES) -> str:
    if not isinstance(label, str):
        raise TypeError("label_to_json expected a [str] but got [" + type(label) + "]")
    if not isinstance(g, GAMES):
        raise TypeError("reformat_json  expected a [GAMES] enum but got [" + type(g) + "]")

    # look up the json object in the deckdrafterprod and return the raw object 
    return json.dumps({
        # "confidence": float(confidence),
        "name": CARD_NAMES[result_index],
        "_id": "", 
        "image_url": PUBLIC_IMAGE_URLS[result_index],
        "tcgplayer_id": TCGPLAYER_IDS[result_index],
        "price": float(LATEST_PRICES[result_index]),
        })


# given a json string (that is raw from the deckdrafterprod), and a Game type
# return a unified json object with the following information:
def reformat_json(json_string : str, g : GAMES) -> str:
    if not isinstance(json_string, str):
        raise TypeError("reformat_json expected a json [str] but got [" + type(json_string) + "]")
    if not isinstance(g, GAMES):
        raise TypeError("reformat_json  expected a [GAMES] enum but got [" + type(g) + "]")

    formatted_json_string = json.dumps({});
    data = json.loads(json_string)

    try: 
        if g == GAMES.LORCANA:
            formatted_json_string = json.dumps({
                'name': data['productName'],
                '_id': data['_id'],
                'image_url': data['images']['large'],
                'tcgplayer_id': data['tcgplayer_productId'], 
                'price': data['price'],
                })
        # elif g == GAMES.MTG:
        # elif g == GAMES.POKEMON:
        else: raise ValueError()
    except KeyError:
        logging.warning('key not found... returning empty json object')
    except ValueError:
        logging.warning('Game Type' + g + 'not supported')
    finally:
        return formatted_json_string
