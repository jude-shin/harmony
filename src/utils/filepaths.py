import os, sys
PROJ_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJ_PATH)


def get_filepaths(cardset, model_name, large_json_name, inital_json_grab):
    data = os.path.join(PROJ_PATH, ".data", "cnn", cardset)
    model = os.path.join(data, model_name)
    keras_model = os.path.join(model, "model.keras")
    checkpoint_model = os.path.join(model, "checkpoint.keras")
    dataset = os.path.join(data, "dataset")
    train_images = os.path.join(dataset, "train_images")
    test_images = os.path.join(dataset, "test_images")
    train_labels = os.path.join(dataset, "train_labels.csv")
    test_labels = os.path.join(dataset, "test_labels.csv")
    formatted_json = os.path.join(
        dataset, f"{large_json_name}({inital_json_grab}).json")
    # I really don't think that I need the init_json_grab, because we are always going to be
    # grabbing all of the images... right?
    # leave it in for now, but you can remove it later
    raw_json = os.path.join(data, "..", "..", f"{large_json_name}.json")
    original = os.path.join(data, "original")
    original_images = os.path.join(original, "images")
    original_labels = os.path.join(original, "labels.csv")
    specs = os.path.join(model, "specs.json")

    filepaths = {
        "DATA": data,
        "MODEL": model,
        "KERAS_MODEL": keras_model,
        "CHECKPOINT_MODEL": checkpoint_model,
        "DATASET": dataset,
        "TRAIN_IMAGES": train_images,
        "TEST_IMAGES": test_images,
        "TRAIN_LABELS": train_labels,
        "TEST_LABELS": test_labels,
        "FORMATTED_JSON": formatted_json,
        "RAW_JSON": raw_json,
        "ORIGINAL": original,
        "ORIGINAL_IMAGES": original_images,
        "ORIGINAL_LABELS": original_labels,
        "SPECS": specs,
    }

    for path in filepaths.values():
        basename = os.path.basename(path)
        name, ext = os.path.splitext(basename)
        if not ext:
            os.makedirs(path, exist_ok=True)
    os.makedirs(model, exist_ok=True)

    return filepaths
