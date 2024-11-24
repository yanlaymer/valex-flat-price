import pandas as pd
import numpy as np
from loguru import logger
from geopy.geocoders import Nominatim, ArcGIS
from scipy.spatial import KDTree
from itertools import combinations
from src.constants import MODEL_COLUMNS
from fuzzywuzzy import fuzz
# Load the analogs data
analogs = pd.read_csv("data/analogs_45K.csv", compression='gzip')

# Calculate 'price_per_square_meter' if not present
if 'price_per_square_meter' not in analogs.columns:
    analogs['price_per_square_meter'] = analogs['price'] / analogs['square']


def get_analog_prices_for_entry(data: pd.DataFrame, entry: dict, required_analogs: int = 3) -> tuple:
    """
    Retrieves analog price statistics and links based on the provided entry.
    """

    # Update the required columns to match your data
    required_columns = [
        "rooms", "year_built", "residential_complex_name", "latitude", "longitude",
        "price_per_square_meter", "link", "building_type", "total_floors",
        "floor", "square", "город"
    ]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logger.error(f"The data is missing required columns: {missing_columns}")
        raise ValueError(f"The data is missing required columns: {missing_columns}")

    # Update the required keys in entry
    required_keys = [
        "rooms", "year_built", "residential_complex_name", "latitude", "longitude",
        "building_type", "total_floors", "floor", "square", "город"
    ]
    missing_keys = [key for key in required_keys if key not in entry]
    if missing_keys:
        logger.error(f"The entry is missing required keys: {missing_keys}")
        raise KeyError(f"The entry is missing required keys: {missing_keys}")

    # Initialize feedback
    feedback = {
        "rooms": False,
        "year_built": False,
        "location": False,
        "building_type": False,
        "total_floors": False,
        "floor": False,
        "square": False,
    }

    # Define the filters and their priorities (lower number is higher priority)
    filters = [
        {"name": "rooms", "function": filter_by_rooms, "priority": 1},
        {"name": "building_type", "function": filter_by_building_type, "priority": 2},
        {"name": "year_built", "function": filter_by_year_built, "priority": 3},
        {"name": "total_floors", "function": filter_by_total_floors, "priority": 4},
        {"name": "floor", "function": filter_by_floor, "priority": 5},
        {"name": "square", "function": filter_by_square, "priority": 6},
    ]

    # Define the filter functions
    def filter_by_rooms(data, entry):
        rooms_number = entry["rooms"]
        return data[data["rooms"] == rooms_number]

    def filter_by_building_type(data, entry):
        building_type = entry["building_type"]
        return data[data["building_type"] == building_type]

    def filter_by_year_built(data, entry, tolerance=5):
        year_built = entry["year_built"]
        year_min = year_built - tolerance
        year_max = year_built + tolerance
        return data[(data["year_built"] >= year_min) & (data["year_built"] <= year_max)]

    def filter_by_total_floors(data, entry):
        total_floors = entry["total_floors"]
        return data[data["total_floors"] == total_floors]

    def filter_by_floor(data, entry):
        flat_floor = entry["floor"]
        total_floors = entry["total_floors"]
        if flat_floor == 1 or flat_floor == total_floors:
            return data[(data["floor"] == 1) | (data["floor"] == data["total_floors"])]
        else:
            return data[(data["floor"] != 1) & (data["floor"] != data["total_floors"])]

    def filter_by_square(data, entry, tolerance=0.15):
        total_square = entry["square"]
        area_min = total_square * (1 - tolerance)
        area_max = total_square * (1 + tolerance)
        return data[(data["square"] >= area_min) & (data["square"] <= area_max)]

    def filter_by_location(data, entry, distance_threshold_km=2):
        city = entry["город"].upper()
        entry_coords_rad = np.radians([entry["latitude"], entry["longitude"]])
        if city in ["АЛМАТЫ", "ШЫМКЕНТ", "АСТАНА"]:
            max_distance = 2 + relaxation_step * 3
        else:
            max_distance = 5 + relaxation_step * 5
        distance_limit_rad = max_distance / 6371.0088
        tree = KDTree(np.radians(data[["latitude", "longitude"]].values))
        indices = tree.query_ball_point(entry_coords_rad.reshape(1, -1), distance_limit_rad)
        indices = indices[0]
        valid_indices = [i for i in indices if i < len(data)]
        return data.iloc[valid_indices]

    # Sort filters by priority
    filters = sorted(filters, key=lambda x: x['priority'])

    # Initialize the list of filters to apply
    active_filters = filters.copy()

    # Start with the full dataset
    analogs = pd.DataFrame()

    # Set maximum number of filters to relax
    max_relaxations = len(filters)

    for relaxation_step in range(max_relaxations + 1):
        filtered_data = data.copy()
        for f in active_filters:
            filtered_data = f['function'](filtered_data, entry)
            if not filtered_data.empty:
                feedback[f['name']] = True
            else:
                feedback[f['name']] = False

        # Apply location filtering
        filtered_data = filter_by_location(filtered_data, entry)
        if not filtered_data.empty:
            feedback["location"] = True
        else:
            feedback["location"] = False

        if len(filtered_data) >= required_analogs:
            logger.info(f"Found {len(filtered_data)} analogs with current filters.")
            analogs = filtered_data
            break
        else:
            logger.info(f"Not enough analogs found ({len(filtered_data)}). Relaxing filters.")
            # Relax the lowest priority filter
            if active_filters:
                active_filters.pop()  # Remove the filter with the lowest priority
            else:
                break  # No filters left to remove

    if analogs.empty:
        logger.warning("No analogs found after relaxing all filters.")
        feedback_status = "RED"
        return (np.nan, np.nan, np.nan, 0, feedback_status, feedback) + tuple([None] * required_analogs)

    # Proceed to finalize analogs as before
    # Ensure 'price_per_square_meter' has valid numeric values
    analogs = analogs.dropna(subset=["price_per_square_meter"])
    if analogs.empty:
        logger.warning("No analogs with valid 'price_per_square_meter' found.")
        feedback_status = "RED"
        return (np.nan, np.nan, np.nan, 0, feedback_status, feedback) + tuple([None] * required_analogs)

    # Compute statistics
    median_price = analogs["price_per_square_meter"].median()
    max_price = analogs["price_per_square_meter"].max()
    min_price = analogs["price_per_square_meter"].min()
    num_analogs = len(analogs)

    # Sort analogs by price
    analogs = analogs.sort_values(by="price_per_square_meter")

    # Remove the 3 lowest-priced properties
    if len(analogs) >= 6:
        analogs = analogs.iloc[3:]
    else:
        logger.warning("Not enough analogs to remove the 3 lowest-priced properties.")

    # Select the 4th, 5th, and 6th properties
    selected_analogs = analogs.iloc[0:3]

    # Extract their links
    analog_links = selected_analogs["link"].tolist()
    while len(analog_links) < required_analogs:
        analog_links.append(None)  # Fill with None if not enough links

    # Determine feedback status
    if all(feedback.values()):
        feedback_status = "GREEN"
    elif any(feedback.values()):
        feedback_status = "ORANGE"
    else:
        feedback_status = "RED"

    logger.info(f"Returning statistics and analog links: Median={median_price}, Max={max_price}, "
                f"Min={min_price}, Count={num_analogs}, Feedback Status={feedback_status}, Links={analog_links}")

    return (median_price, max_price, min_price, num_analogs, feedback_status, feedback) + tuple(analog_links)


def get_location(city, district, street, house_number, housing_comlex_name):
    geolocator = Nominatim(user_agent="another_app")
    geolocator_v2 = ArcGIS(user_agent="fallback_app")

    # make all letters uppercase
    city = city.upper()
    district = district.upper()
    street = street.upper()
    house_number = house_number.upper()
    housing_comlex_name = (
        housing_comlex_name.upper()
        if len(housing_comlex_name) > 0 or housing_comlex_name
        else "НЕТ"
    )

    if "МКР" in street:
        street = street.replace("МКР", "МИКРОРАЙОН")

    if "МКР" in district:
        district = district.replace("МКР", "МИКРОРАЙОН")

    if "Р-Н" in district:
        district = district.replace("Р-Н", "РАЙОН")

    if ("Р-Н" not in district) or ("РАЙОН" not in district):
        district = f"{district} РАЙОН"

    components = {
        "city": city,
        "district": district,
        "street": street,
        "house_number": house_number,
        "housing_comlex_name": housing_comlex_name,
    }

    all_combinations = []
    sorted_components = sorted(components.items(), key=lambda x: -len(x[1]))
    for r in range(len(sorted_components), 0, -1):
        for subset in combinations(sorted_components, r):
            address = ", ".join(item[1] for item in subset)
            all_combinations.append(address)

    # Loop through the generated combinations
    tries = len(all_combinations)
    logger.info(f"Trying {tries} combinations")
    inc = 0
    for address in all_combinations:
        try:
            logger.info(f"Trying geolocator NOMINATIM")
            location = geolocator.geocode(address)
            inc += 1
            if location:
                logger.info(
                    f"Tried {inc} combinations using Nominatim. Found location by address: {address}"
                )
                return location
        except:
            logger.error("Nominatim failed. Switching to ArcGIS")
            inc += 1
            # If Nominatim fails, switch to ArcGIS
            location = geolocator_v2.geocode(address)
            if location:
                logger.info(
                    f"Tried {inc} combinations using ArcGIS. Found location by address: {address}"
                )
                return location

    return None

def get_flat_features(entry: pd.Series) -> tuple:
    """
    Returns a Series with the features of the flat, feedback status, and feedback dictionary.
    """
    logger.info("FEATURE EXTRACTION")
    logger.info(f"INITIAL ENTRY: {entry}")
    try:
        city = entry["city"].upper() if entry["city"] else ""
        district = entry["district"].upper() if entry["district"] else ""
        street = entry["street"].upper() if entry["street"] else ""
        house_number = entry["home_number"].upper() if entry["home_number"] else ""
    except AttributeError:
        city = ""
        district = ""
        street = ""
        house_number = ""

    residential_complex_name = (
        entry["residential_complex"].upper()
        if entry["residential_complex"] is not None
        else ""
    )
    total_square_meters = entry["total_square"]
    building_type = (
        entry["building_type"].upper()
        if entry["building_type"] != "НЕИЗВЕСТНЫЙ"
        else "ИНОЕ"
    )
    floor = entry["flat_floor"]
    total_floors = entry["building_floor"]
    rooms_number = entry["live_rooms"]
    year_built = entry["building_year"]
    bathroom = entry["flat_toilet"].upper()
    former_hostel = True if entry["flat_priv_dorm"] == "Да" else False

    location = None
    if entry["latitude"] is not None and entry["longitude"] is not None:
        latitude = entry["latitude"]
        longitude = entry["longitude"]
        logger.info(f"Latitude and Longitude provided: {latitude}, {longitude}")
        location = Nominatim(user_agent="my_app").reverse(f"{latitude}, {longitude}")
        logger.info(f"Location Found: {location.address}")
    else:
        location = get_location(
            city, district, street, house_number, residential_complex_name
        )
        if location is None:
            logger.error("Could not find location for the flat.")
            return None, "RED", {}
        logger.info(f"Location Found: {location.address}")

    latitude = location.latitude
    longitude = location.longitude

    residential_complex_name = "НЕТ" if residential_complex_name == "" else residential_complex_name

    entry_series = pd.Series(
        [
            latitude,
            longitude,
            floor,
            total_floors,
            rooms_number,
            total_square_meters,
            year_built,
            building_type,
            residential_complex_name,
            bathroom,
            former_hostel,
            None,  # analog_prices_median placeholder
            None,  # analog_prices_max placeholder
            None,  # analog_prices_min placeholder
            city,
        ],
        index=MODEL_COLUMNS + ['город'],
    )

    logger.info(f"ENTRY: {entry_series}")

    logger.info("SEARCHING ANALOGS")
    analogs_data = get_analog_prices_for_entry(analogs, entry_series)

    (
        entry_series["analog_prices_median"],
        entry_series["analog_prices_max"],
        entry_series["analog_prices_min"],
        entry_series["analogs_found"],
        feedback_status,
        feedback,
        entry_series["analog_1"],
        entry_series["analog_2"],
        entry_series["analog_3"],
    ) = analogs_data

    if location:
        entry_series['address_geocoder'] = location.address
    else:
        entry_series['address_geocoder'] = "Не найдено"

    logger.info(f"ENTRY AFTER ANALOGS: {entry_series}")

    return entry_series, feedback_status, feedback
