import pandas as pd
import numpy as np
from loguru import logger
from geopy.geocoders import Nominatim
from scipy.spatial import KDTree
from itertools import combinations
from src.constants import MODEL_COLUMNS

analogs = pd.read_csv("data/analogs.csv", compression='gzip')

def get_analog_prices_for_entry(data, entry):
    # Step 1: Filter based on rooms_number and construction_year
    mask = (data['rooms_number'] == entry['rooms_number']) & \
           (np.abs(data['construction_year'] - entry['construction_year']) <= 5)
    filtered_data = data[mask]
    
    # Step 2: Check for the same housing_complex_name
    analogs = None
    if entry['housing_comlex_name'] != 'None':
        entry['housing_comlex_name'] = entry['housing_comlex_name'].upper()
        mask = (filtered_data['housing_comlex_name'] == entry['housing_comlex_name'])
        analogs = filtered_data[mask]
    
    
    # Step 3: If no sufficient analogs found using housing_complex_name, perform spatial search
    if analogs is None or len(analogs) < 5:
        tree = KDTree(np.radians(filtered_data[['latitude', 'longitude']].values))
        distance_limit_rad = 1 / 6371.0088
        _, indices = tree.query([np.radians(entry['latitude']), np.radians(entry['longitude'])], distance_upper_bound=distance_limit_rad, k=3)
        
        # Ensure indices is always a list
        indices = np.atleast_1d(indices)
        
        analogs = filtered_data.iloc[indices]
        
        if analogs is None or len(analogs) < 3:
            
            tree = KDTree(np.radians(filtered_data[['latitude', 'longitude']].values))
            distance_limit_rad = 3 / 6371.0088
            _, indices = tree.query([np.radians(entry['latitude']), np.radians(entry['longitude'])], distance_upper_bound=distance_limit_rad, k=5)
            
            analogs = filtered_data.iloc[indices]
            
        
    
    # Return the statistics for the found analogs
    return analogs['price_per_square_meter'].median(), analogs['price_per_square_meter'].max(), analogs['price_per_square_meter'].min(), len(analogs), analogs['link'].tolist()[0], analogs['link'].tolist()[1], analogs['link'].tolist()[2]

def get_location(city, district, street, house_number, housing_comlex_name):
    geolocator = Nominatim(user_agent="my_app")
    
    # make all letters uppercase
    city = city.upper()
    district = district.upper()
    street = street.upper()
    house_number = house_number.upper()
    housing_comlex_name = housing_comlex_name.upper()
    
    components = {
        'city': city,
        'district': district,
        'street': street,
        'house_number': house_number,
        'housing_comlex_name': housing_comlex_name
    }
    
    # Generate all possible combinations of address components
    all_combinations = []
    for r in range(len(components), 0, -1):
        for subset in combinations(components.keys(), r):
            address = ', '.join(components[key] for key in subset)
            all_combinations.append(address)

    # Loop through the generated combinations
    for address in all_combinations:
        location = geolocator.geocode(address)
        if location:
            return location

    return None

        # data = {
        #     "city": city,
        #     "district": district,
        #     "street": street,
        #     "residential_complex": residential_complex,
        #     "home_number": home_number,
        #     "building_type": building_type,
        #     "total_square": total_square,
        #     "kitchen_square": kitchen_square,
        #     "flat_floor": flat_floor,
        #     "building_floor": building_floor,
        #     "live_rooms": live_rooms,
        #     "building_year": building_year,
        #     "flat_priv_dorm": flat_priv_dorm,
        #     "flat_renovation": flat_renovation,
        #     "flat_toilet": flat_toilet,
        #     "live_furniture": live_furniture
        # }

MODEL_COLUMNS = ['latitude', 'longitude', 'floor', 'floors_number', 'rooms_number',
                 'total_square_meters', 'construction_year', 'wall_type',
                 'housing_comlex_name', 'bathroom', 'former_hostel',
                 'analog_prices_median', 'analog_prices_max', 'analog_prices_min']


def get_flat_features(entry: pd.Series) -> pd.Series:
    """
    Returns a Series with the features of the flat.
    """
    logger.info("FEATURE EXTRACTION")
    logger.info(f"INITIAL ENTRY: {entry}")
    try:
        city = entry['city'].upper()
        district = entry['district'].upper()
        street = entry['street'].upper()
        house_number = entry['home_number'].upper()
    except AttributeError:
        city = None
        district = None
        street = None
        house_number = None
    
        
    housing_comlex_name = entry['residential_complex'].upper() if entry['residential_complex'] is not None else "НЕТ"
    total_square_meters = entry['total_square']
    wall_type = entry['building_type'].upper() if entry['building_type'] != "НЕИЗВЕСТНЫЙ" else "ИНОЕ"
    floor = entry['flat_floor']
    floors_number = entry['building_floor']
    rooms_number = entry['live_rooms']
    construction_year = entry['building_year']
    bathroom = entry['flat_toilet'].upper()
    former_hostel = True if entry['flat_priv_dorm'] == 'Да' else False
    
    if entry['latitude'] is not None and entry['longitude'] is not None:
        latitude = entry['latitude']
        longitude = entry['longitude']
        logger.info(f"Latitude and Longitude provided: {latitude}, {longitude}")
        # address
        location = Nominatim(user_agent="my_app").reverse(f"{latitude}, {longitude}")
        logger.info(f"Location Found: {location.address}")
    else:
        location = get_location(city, district, street, house_number, housing_comlex_name)
        if location is None:
            logger.error("Could not find location for the flat.")
            return None
        logger.info(f"Location Found: {location.address}")
    
    latitude = location.latitude
    longitude = location.longitude
    analog_prices_median=None
    analog_prices_max=None
    analog_price_min=None
    
    entry = pd.Series([latitude, 
                       longitude, 
                       floor, 
                       floors_number, 
                       rooms_number, 
                       total_square_meters, 
                       construction_year, 
                       wall_type, 
                       housing_comlex_name, 
                       bathroom, 
                       former_hostel, 
                       analog_prices_median, 
                       analog_prices_max,
                       analog_price_min], index=MODEL_COLUMNS)
    
    logger.info(f"ENTRY: {entry}")
    
    logger.info("SEARCHING ANALOGS")
    analog_prices_median, analog_prices_max, analog_prices_min, analogs_number, analog_link_1, analog_link_2, analog_link_3 = get_analog_prices_for_entry(analogs, entry)
    
    entry['analog_prices_median'], entry['analog_prices_max'], entry['analog_prices_min'], entry['analogs_found'], entry['analog_1'], entry['analog_2'], entry['analog_3'] = get_analog_prices_for_entry(analogs, entry)
    
    logger.info(f"ENTRY AFTER ANALOGS: {entry}")
    
    return entry
    