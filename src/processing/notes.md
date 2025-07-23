# holds everything data related:
- processing deckdrafterprod.json files
- artificial dataset expansion (image manipulation)
    - turn this into a tf.data.Dataset
- making config.toml files
- queue for images that are to be trained (this may be in the tf.data.Dataset)
- exporting the .keras files from the keras_models folder to the saved_models folder
