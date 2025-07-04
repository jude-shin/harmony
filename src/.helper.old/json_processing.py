import shutil
import json

from copy import deepcopy


# ==================================================
# HELPERS

def create_smaller_json(
    json_filepath: str, new_filepath: str, image_count: int, verbose: bool = True
):
    if verbose:
        print(f"Copying {image_count} Objects ...\n")

    # # Create a new file path for the smaller JSON file
    # d, f = json_filepath.rsplit('/', 1)
    # f = f.replace(".json", f"_small({image_count}).json")

    # new_filepath = os.path.join(model_filepath, f)

    # Load the entire JSON file
    with open(json_filepath, "r", encoding="utf-8") as original_file:
        data = json.load(original_file)

    # get the specified # of data from the dataset
    if image_count == -1:
        if verbose:
            print(f'Copying ALL objects from "{
                  json_filepath}" to "{new_filepath}" ...')
        small_data = data
    else:
        if verbose:
            print(
                f'Copying {image_count} objects from "{
                    json_filepath}" to "{new_filepath}" ...'
            )
        small_data = data[:image_count]

    # Write the small data to the new JSON file
    with open(new_filepath, "w", encoding="utf-8") as new_file:
        json.dump(small_data, new_file, indent=4)

    # Copy the original file's permissions to the new file
    shutil.copymode(json_filepath, new_filepath)

    if verbose:
        print("\nFinished Copying!")


def filter_attributes_json(
    json_filepath: str, attributes: list[str], verbose: bool = True
):
    if verbose:
        print(f"Filtering {json_filepath} with only {attributes} ...")
    # Open the original JSON file and load the data
    with open(json_filepath, "r", encoding="utf-8") as original_file:
        data = json.load(original_file)

    # Filter the objects to only include the specified attributes
    filtered_data = []

    for obj in data:
        new_obj = {}
        for attr in attributes:
            if attr in obj:
                new_obj[attr] = obj[attr]

        if "card_faces" in obj:
            for face in obj["card_faces"]:
                face_obj = deepcopy(new_obj)
                for attr in attributes:
                    if attr in face:
                        face_obj[attr] = face[attr]
                filtered_data.append(face_obj)
            continue
        filtered_data.append(new_obj)

    # Write the filtered data back to the original JSON file
    with open(json_filepath, "w", encoding="utf-8") as original_file:
        json.dump(filtered_data, original_file, indent=4)

    if verbose:
        print("Finished filtering!")


def format_image_attributes(
    json_filepath: str,
    image_size: str,
    image_attribute_label: str,
    verbose: bool = True,
):
    if verbose:
        print(f"Formatting {json_filepath} with {image_size} image size")

    # Load the JSON file
    with open(json_filepath, "r") as f:
        data = json.load(f)

    filtered_data = []

    # Add the attribute to each dictionary
    for json_object in data:
        if (
            image_attribute_label in json_object
            # use the first images that we see (these would probably be the best)
        ):
            json_object["image"] = json_object[image_attribute_label][image_size]
            # delete the attribute that the json object has of the old image data
            del json_object[image_attribute_label]
            filtered_data.append(json_object)

        else:
            # if there is no image found for the object, just skip it for now, and print a message
            if verbose:
                print(
                    f"({json_object['_id']}) NO IMAGES FOUND [removed from json] ...")

    # Write the modified data back to the JSON file
    with open(json_filepath, "w") as f:
        json.dump(filtered_data, f, indent=4)

    if verbose:
        print("Finished formatting!")


def encode_alphanumeric_to_int(json_filepath: str):
    # Load the JSON file
    with open(json_filepath, "r") as f:
        data = json.load(f)

    # Assign the index of each object in the JSON file as the 'encoded' attribute
    for index, json_object in enumerate(data):
        if "_id" in json_object:
            # Use the index as the 'encoded' value
            json_object["encoded"] = str(index)

    # Write the modified data back to the JSON file
    with open(json_filepath, "w") as f:
        json.dump(data, f, indent=4)


# ==================================================


def format_json(
    raw_json_filepath: str,
    new_filepath: str,
    image_count: int,
    image_size: str,
    attributes: list[str] = ["_id", "productUrlName", "types"], # TODO: remove the productUrl name when this project is done
    verbose: bool = True,
):
    image_attribute_label = "images"

    if verbose:
        print("\n--- CREATING SEPERATE JSON ---")
    create_smaller_json(
        json_filepath=raw_json_filepath,
        new_filepath=new_filepath,
        image_count=image_count,
    )

    if verbose:
        print("\n--- FILTERING JSON ---")
    attributes.append(image_attribute_label)
    filter_attributes_json(json_filepath=new_filepath, attributes=attributes)

    if verbose:
        print("\n--- FORMATTING JSON ATTRIBUTES ---")
    format_image_attributes(
        json_filepath=new_filepath,
        image_size=image_size,
        image_attribute_label=image_attribute_label,
    )

    if verbose:
        print("\n--- JSON FULLY FORMATTED ---\n")
