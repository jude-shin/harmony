something = {
        'hello': 12,
        'there': {'one': 1, 'two': 22, 'three': 333},
        'worlds': 'shoot',
        }

# config = load_model_config(pl.LORCANA)
# config = config['m0']
# keys = config.get['deckdrafterprod_img_keys']
keys = ['there', 'two']

url = something

for k in keys:
    url = url.get(k)
    
print(url)
