import os, sys
PROJ_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJ_PATH)

from helper.image_processing import get_tensor_from_image, get_tensor_from_dir
from helper.helper import get_elapsed_time
from tensorflow.keras import callbacks, layers, models, optimizers, mixed_precision  # type: ignore
import numpy as np
import json
import pandas as pd
import time
id_to_types= {
    0: "None",
    1: "Grass",
    2: "Lightning", 
    3: "Darkness",
    4: "Fairy", 
    5: "Fire", 
    6: "Psychic",
    7: "Metal",
    8: "Dragon",
    9: "Water",
    10: "Fighting",
    11: "Colorless",
}

def predict_folder(model_path, overall_json_path, img_folder_path):
    st = time.time()

    model = models.load_model(os.path.join(model_path, 'model.keras'))
    predictions = []
    images = [img for img in os.listdir(
        img_folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # play with this to remove the need for the metadata file
    # this gives the input layer size of the model
    _, img_width, img_height, _ = model.input_shape

    for i, image_name in enumerate(images):
        img_path = os.path.join(img_folder_path, image_name)
        img_tensor = get_tensor_from_dir(
            img_path, img_width=img_width, img_height=img_height)

        # make the image tensor match the input layer of the model
        img_tensor = np.expand_dims(img_tensor, axis=0)
        prediction = model.predict(img_tensor)
        prediction, confidence = np.argmax(
            prediction), prediction[0, np.argmax(prediction)]
        print(f'Image: {image_name}')
        print(f'Prediction: {prediction}')
        print(f'Confidence: {confidence}')

        csv_path = os.path.join(model_path, 'test_labels.csv')
        card_info_df = pd.read_csv(csv_path)

        predicted_id = card_info_df[card_info_df['label']
                                    == prediction]['_id'].iloc[0]
        predicted_obj = find_object_by_id(overall_json_path, predicted_id)
        if predicted_obj is not None:
            predicted_name = predicted_obj['productUrlName']
        else:
            predicted_name = None
        # predictions[i][image_name] = {'_id': predicted_id, 'productUrlName': predicted_name, 'confidence': str(confidence)}
        predictions.append({image_name: {
                           '_id': predicted_id, 'productUrlName': predicted_name, 'confidence': str(confidence)}})

    overall_predict_time = get_elapsed_time(st)
    # overall_predict_time/len(images)
    ave_time_per_card = (time.time()-st)/len(images)

    with open(os.path.join(img_folder_path, 'info.txt'), 'a') as f:
        f.write(f'Overall Prediction Time: {overall_predict_time}\n')
        f.write(f'Averate Time Per Card: {ave_time_per_card}\n')
        f.write(f'# of Cards: {len(images)}\n')

    predictions_json_path = os.path.join(
        img_folder_path, f'{os.path.basename(model_path)}_predictions.json')
    with open(predictions_json_path, 'w') as file:
        json.dump(predictions, file, indent=4)

    print('\n')
    print(f'Overall Prediction Time: {overall_predict_time}\n')
    print(f'Averate Time Per Card: {ave_time_per_card}\n')
    print(f'# of Cards: {len(images)}\n')

def predict_image(img_folder_path, image_name, img_width, img_height, model):
        img_path = os.path.join(img_folder_path, image_name)
        img_tensor = get_tensor_from_dir(
            img_path, img_width=img_width, img_height=img_height)

        img_tensor = np.expand_dims(img_tensor, axis=0)
        prediction = model.predict(img_tensor)
        prediction, confidence = np.argmax(
            prediction), prediction[0, np.argmax(prediction)]
        
        return prediction, confidence

def predict_folder_two_link(
        overall_model_path,
        smaller_models,
        overall_json_path, 
        img_folder_path,   
        csv_path,
): 
    predictions = []
    subtimes = []
    st = time.time()

    overall_model = models.load_model(os.path.join(overall_model_path, 'model.keras'))

    images = [img for img in os.listdir(
        img_folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    _, img_width, img_height, _ = overall_model.input_shape
    card_info_df = pd.read_csv(csv_path)

    for image_name in images:
        sub_start_time = time.time()

        sub_type, sub_type_confidence= predict_image(
            img_folder_path=img_folder_path,
            image_name=image_name,
            img_width=img_width,
            img_height=img_height,
            model=overall_model,
        )
        sub_model = models.load_model(os.path.join(smaller_models[int(sub_type)], 'model.keras'))

        final_prediction, final_prediction_confidence = predict_image(
            img_folder_path=img_folder_path,
            image_name=image_name,
            img_width=img_width,
            img_height=img_height,
            model=sub_model,
        )

        predicted_id = card_info_df[card_info_df['label'] == final_prediction]['_id'].iloc[0]

        sub_end_time = time.time()
        raw_time = sub_end_time - sub_start_time
        subtimes.append(raw_time)
        

        print(f'Image: {image_name}')
        print(f'FIRST prediction value: {sub_type}')
        print(f'FIRST prediction _id: {id_to_types[sub_type]}')
        print(f'FIRST confidence: {sub_type_confidence}\n')

        print(f'FINAL prediction value: {final_prediction}')
        print(f'FINAL prediction _id: {predicted_id}')
        print(f'FINAL confidence: {final_prediction_confidence}\n')

        print(f'OVERALL Confidence: {final_prediction_confidence*sub_type_confidence}')
        print(f'raw prediction time: {raw_time}')

        # predicted_obj = find_object_by_id(overall_json_path, predicted_id)
        # if predicted_obj is not None:
        #     predicted_name = predicted_obj['productUrlName']
        # else:
        #     predicted_name = None

        predictions.append({image_name: {
                        'first prediction value': str(sub_type),
                        'first predictin _id': str(id_to_types[sub_type]),
                        'first confidence': str(sub_type_confidence),
                        'FINAL prediction value': str(final_prediction),
                        'FINAL prediction _id': str(predicted_id),
                        'FINAL confidence': str(final_prediction_confidence),
                        'overall _id': str(predicted_id), 
                        'overall confidence': str(final_prediction_confidence*sub_type_confidence),
                        'raw prediction time': str(raw_time)
        }})

    overall_predict_time = get_elapsed_time(st)
    # overall_predict_time/len(images)
    ave_time_per_card = (time.time()-st)/len(images)

    with open(os.path.join(img_folder_path, 'info.txt'), 'a') as f:
        f.write(f'Overall Prediction Time: {overall_predict_time}\n')
        f.write(f'Averate Time Per Card: {ave_time_per_card}\n')
        f.write(f'# of Cards: {len(images)}\n')

    predictions_json_path = os.path.join(
        img_folder_path, f'{os.path.basename(overall_model_path)}_predictions.json')
    with open(predictions_json_path, 'w') as file:
        json.dump(predictions, file, indent=4)

    print('\n')
    print(f'Overall Prediction Time: {overall_predict_time}\n')
    ave = 0
    for n in subtimes: ave += n 
    ave = ave / len(subtimes)
    print(f'Average Raw Time per card: {ave}\n')
    print(f'# of Cards: {len(images)}\n')

    print("RESULTS SAVED AT THE LARGER MODEL PATH")
        





def find_object_by_id(overall_json_path, target_id):
    # Assuming overall_json_path is a string containing the path to a JSON file
    with open(overall_json_path, 'r') as file:
        data = json.load(file)

    for obj in data:
        if str(obj['_id']) == str(target_id):
            return obj
    return None

if __name__ == '__main__':
    smaller_models = {
        # 0: '/home/jude/harmony_org/card_recognition_ml/.data/cnn/Pokemon0Card/POKEMON0_2024.09.16.17.10.35/',
        0: '/home/jude/harmony_org/card_recognition_ml/.data/cnn/Pokemon0Card/POKEMON0_2024.09.18.04.20.08/',
        
        1: '/home/jude/harmony_org/card_recognition_ml/.data/cnn/Pokemon1Card/POKEMON1_2024.09.17.20.46.38',
        2: '/home/jude/harmony_org/card_recognition_ml/.data/cnn/Pokemon2Card/POKEMON2_2024.09.17.20.59.29',
        3: '/home/jude/harmony_org/card_recognition_ml/.data/cnn/Pokemon3Card/POKEMON3_2024.09.17.21.26.11',
        4: '/home/jude/harmony_org/card_recognition_ml/.data/cnn/Pokemon4Card/POKEMON4_2024.09.17.21.30.30',
        5: '/home/jude/harmony_org/card_recognition_ml/.data/cnn/Pokemon5Card/POKEMON5_2024.09.17.21.35.59',
        6: '/home/jude/harmony_org/card_recognition_ml/.data/cnn/Pokemon6Card/POKEMON6_2024.09.17.21.58.23',
        7: '/home/jude/harmony_org/card_recognition_ml/.data/cnn/Pokemon7Card/POKEMON7_2024.09.17.22.12.14',
        8: '/home/jude/harmony_org/card_recognition_ml/.data/cnn/Pokemon8Card/POKEMON8_2024.09.17.22.14.50',
        9: '/home/jude/harmony_org/card_recognition_ml/.data/cnn/Pokemon9Card/POKEMON9_2024.09.17.22.18.36',
        10:'/home/jude/harmony_org/card_recognition_ml/.data/cnn/Pokemon10Card/POKEMON10_2024.09.17.22.29.53',
        11:'/home/jude/harmony_org/card_recognition_ml/.data/cnn/Pokemon11Card/POKEMON11_2024.09.18.03.24.08',
    }
     
    predict_folder_two_link(
        overall_model_path='/home/jude/harmony_org/card_recognition_ml/.data/cnn/PokemonCard/POKEMON_2024.09.18.23.43.29/',
        smaller_models=smaller_models,
        img_folder_path='/home/jude/harmony_org/scans/pokemon/card_5',
        overall_json_path='/home/jude/harmony_org/card_recognition_ml/.data/cnn/PokemonCard.old/dataset/deckdrafterprod.PokemonCard(-1).json',
        csv_path='/home/jude/harmony_org/card_recognition_ml/.data/cnn/PokemonCard.old/dataset/test_labels.csv'

    )



# if __name__ == '__main__':
#     model_path = '/home/jude/Work/Store Pass/card_recognition_ml/.data/cnn/ONEPIECE_0.0.0'
#     img_folder_path = '/home/jude/Work/Store Pass/card_recognition_ml/.data/cnn/ONEPIECE_0.0.0/train_images'
#     overall_json_path = '/home/jude/Work/Store Pass/card_recognition_ml/.data/deckdrafterprod.OnePieceCard.json'

#     predict_folder(model_path=model_path,
#                    overall_json_path=overall_json_path, img_folder_path=img_folder_path)
