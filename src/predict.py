from datetime import date
import numpy as np
from pickle import load
from loguru import logger

ARTEFACTS_PATH = 'model/artefacts/'
ENCODERS = {
    "district": "district_encoder.sav",
    "owner": "owner_encoder.sav",
    "building": "building_encoder.sav",
    "flat_renovation": "flat_renovation_encoder.sav",
    "flat_priv_dorm": "flat_priv_encoder.sav",
    "furniture": "furniture_encoder.sav",
    "toilet": "toilet_encoder.sav"
}

encoders = {}

def transform_with_logging(encoder, value, encoder_name):
    if value is None:
        logger.error(f"{encoder_name} received a None value.")
        raise ValueError(f"{encoder_name} received a None value.")
    try:
        return encoder.transform([value])[0]
    except ValueError as e:
        logger.error(f"Error transforming with {encoder_name}. Value: {value}. Error: {str(e)}")
        raise

for key, filename in ENCODERS.items():
    with open(ARTEFACTS_PATH + filename, 'rb') as f:
        encoders[key] = load(f)

with open('model/rf_regressor.sav', 'rb') as f:
    rf_regressor = load(f)


def calculate_first_floor(flat_floor):
    return 1 if flat_floor == 1 else 0


def calculate_last_floor(flat_floor, building_floor):
    return 1 if flat_floor == building_floor else 0


def calculate_building_age(building_year):
    return date.today().year - building_year


def calculate_square_per_room(total_square, live_rooms):
    return total_square / live_rooms

def get_flat_price(json_to_send):
    print(json_to_send)
    try:
        is_first_floor = calculate_first_floor(json_to_send.get("flat_floor"))
        is_last_floor = calculate_last_floor(json_to_send.get("flat_floor"), json_to_send.get("building_floor"))
        square_per_room = calculate_square_per_room(json_to_send.get("total_square"), json_to_send.get("live_rooms"))
        building_age = calculate_building_age(json_to_send.get("building_year"))

        district = transform_with_logging(encoders["district"], json_to_send.get('district'), "district")
        owner = transform_with_logging(encoders["owner"], "Хозяин недвижимости", "owner")
        building_type = transform_with_logging(encoders["building"], json_to_send.get('building_type'), "building")
        flat_priv_dorm = transform_with_logging(encoders["flat_priv_dorm"], json_to_send.get('flat_priv_dorm'), "flat_priv_dorm")
        flat_renovation = transform_with_logging(encoders["flat_renovation"], json_to_send.get('flat_renovation'), "flat_renovation")
        flat_toilet = transform_with_logging(encoders["toilet"], json_to_send.get('flat_toilet'), "toilet")
        live_furniture = transform_with_logging(encoders["furniture"], json_to_send.get('live_furniture'), "furniture")
        

        to_model = np.array([
            district, json_to_send.get('live_rooms'), owner, json_to_send.get('total_square'), json_to_send.get('kitchen_square'),
            json_to_send.get('flat_floor'), json_to_send.get('building_floor'), building_age, building_type, flat_priv_dorm,
            flat_renovation, flat_toilet, live_furniture, is_first_floor, is_last_floor, square_per_room
        ])
        
        logger.info(f"to_model: {to_model}")

        prediction = rf_regressor.predict(to_model.reshape(1, -1))
        factor = 1.0
        if json_to_send.get("city") == "Алматы":
            factor = np.exp(0.20) if len(json_to_send.get("residential_complex")) > 0 else np.exp(0.09)
        elif json_to_send.get("city", "city") in ["Шымкент", "Астана", "Павлодар"]:
            factor = 0.8 * (np.exp(0.10) if len(json_to_send.get("residential_complex")) > 0 else 1)
            if json_to_send.get("district") == "Байконур":
                factor *= 0.8
        elif json_to_send.get("city") == "":
            factor = 0.07
        else:
            factor = 0.35

        return {
            "price": prediction[0] * 10e5 * factor * json_to_send.get('total_square'),
            "message": f"Price calculated for {json_to_send.get('city')}, {json_to_send.get('district')}, {json_to_send.get('street')} {json_to_send.get('home_number')}",
            "status_code": 200
        }

    except ValueError as e:
        logger.error(e)
        return {
            "price": 0,
            "message": f"Flat located at {json_to_send.get('city')}, {json_to_send.get('district')}, {json_to_send.get('street')} {json_to_send.get('home_number')} couldn't be estimated. Error: {e}",
            "status_code": 400
        }

