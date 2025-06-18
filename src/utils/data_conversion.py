from config.constants import Games

# label: what the tensorflow model will spit out
# _id: tcg/deckdrafterprod unique _id
#   allows us to get more information about the card


# given a label (string)
# return the _id (string) that is associated with that label
def label_to_id(label : str) -> str:
    if not isinstance(label, str):
        raise TypeError("label_to_id expected a [str] but got [" + type(label) + "]")

    json = label_to_json(label)

    # extract only the _id from the json object
    return ""

# given an _id (string)
# return the label (string) that is associated with that _id
def id_to_label(_id : str) -> str:
    if not isinstance(_id, str):
        raise TypeError("id_to_label expected a [str] but got [" + type(_id) + "]")

    # look up the variable
    return ""

# given a label (string)
# return a json object with certain fields
def label_to_json(label : str) -> str:
    if not isinstance(label, str):
        raise TypeError("label_to_json expected a [str] but got [" + type(label) + "]")

    # look up the json object in the deckdrafterprod and return the raw object 
    return jsonify({
        # "confidence": float(confidence),
        "name": CARD_NAMES[result_index],
        "_id": "", 
        "image_url": PUBLIC_IMAGE_URLS[result_index],
        "tcgplayer_id": TCGPLAYER_IDS[result_index],
        "price": float(LATEST_PRICES[result_index]),
        })


# given a json string (that is raw from the deckdrafterprod), and a Game enum
# return a unified json object with the following information:
def reformat_json(json_string : str, t : Games) -> str:
    # switch case on the type to determine what format the tcg player is in
    # based on that fill in the new json object with the corrected fields
    if t == Games.MTG:
        return "we found mtg"
    elif t == Games.LORCANA:
        return "we found lorcana"
    elif t == Games.POKEMON:
        return "we found pokemon"
    else: 
        print("something went wrong")
        return ""
