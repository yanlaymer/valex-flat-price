import pandas as pd
import numpy as np
from loguru import logger
from geopy.geocoders import Nominatim, ArcGIS, GoogleV3
from scipy.spatial import KDTree
from itertools import combinations
from src.constants import MODEL_COLUMNS
from fuzzywuzzy import fuzz

analogs = pd.read_csv("data/current_analogs_40K.csv") # db in rl


def get_analog_prices_for_entry(data: pd.DataFrame, entry: dict, required_analogs: int = 2) -> tuple:
    """
    Retrieves analog price statistics and links based on the provided entry.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing analogs with necessary features.
    - entry (dict): A dictionary containing details of the entry to find analogs for.
    - required_analogs (int): The number of analog links to return. Default is 3.

    Returns:
    - tuple: A tuple containing median, max, min price per square meter, number of analogs,
             and a list of analog links (with length equal to required_analogs).
    """
    # Validate required columns in data
    required_columns = ["rooms_number", "construction_year", "housing_comlex_name", "latitude", "longitude",
                        "price_per_square_meter", "link"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logger.error(f"The data is missing required columns: {missing_columns}")
        raise ValueError(f"The data is missing required columns: {missing_columns}")

    # Validate required keys in entry
    required_keys = ["rooms_number", "construction_year", "housing_comlex_name", "latitude", "longitude"]
    missing_keys = [key for key in required_keys if key not in entry]
    if missing_keys:
        logger.error(f"The entry is missing required keys: {missing_keys}")
        raise KeyError(f"The entry is missing required keys: {missing_keys}")

    # Step 1: Filter based on rooms_number
    try:
        rooms_number = entry["rooms_number"]
        mask = (-1 <= data["rooms_number"] - rooms_number <= 1)
        filtered_data = data[mask].copy()
        logger.info(f"Filtered data based on rooms_number={rooms_number}: {filtered_data.shape[0]} records found.")
    except Exception as e:
        logger.exception("Error filtering data based on rooms_number.")
        raise e

    # Step 2: Fuzzy match on housing_complex_name if applicable
    analogs = pd.DataFrame()
    housing_complex_name = entry.get("housing_comlex_name", "").strip().upper()

    if housing_complex_name and housing_complex_name != "NONE":
        try:
            threshold = 85  # Define fuzzy matching threshold
            scores = filtered_data["housing_complex_name"].fillna("").str.upper().apply(
                lambda x: fuzz.token_set_ratio(x, housing_complex_name)
            )
            filtered_data_with_scores = filtered_data.assign(score=scores)
            analogs = filtered_data_with_scores[filtered_data_with_scores['score'] >= threshold]
            analogs = analogs.sort_values(by='score', ascending=False)
            logger.info(f"Found {len(analogs)} analogs by housing_complex_name with name '{housing_complex_name}'.")
        except Exception as e:
            logger.exception("Error during fuzzy matching of housing_complex_name.")

    # Step 3: If not enough analogs found, use geographic proximity
    if len(analogs) < required_analogs:
        try:
            # Define distance thresholds in kilometers
            distance_thresholds = [1, 3, 10, 15, 20, 50]  # You can adjust these values as needed
            entry_coords_rad = np.radians([entry["latitude"], entry["longitude"]])
            tree = KDTree(np.radians(filtered_data[["latitude", "longitude"]].values))

            for threshold_km in distance_thresholds:
                distance_limit_rad = threshold_km / 6371.0088  # Earth's radius in kilometers
                distances, indices = tree.query(entry_coords_rad.reshape(1, -1), 
                                                distance_upper_bound=distance_limit_rad, 
                                                k=required_analogs)

                # Flatten the indices and filter out invalid entries
                indices = indices.flatten()
                valid_indices = indices[indices < len(filtered_data)]
                new_analogs = filtered_data.iloc[valid_indices]

                # If fuzzy matched analogs exist, combine them
                if not analogs.empty:
                    new_analogs = pd.concat([analogs, new_analogs]).drop_duplicates()

                if len(new_analogs) >= required_analogs:
                    analogs = new_analogs.head(required_analogs)
                    logger.info(f"Found {len(analogs)} analogs within {threshold_km} km.")
                    break
                else:
                    logger.info(f"Only found {len(new_analogs)} analogs within {threshold_km} km. Expanding search...")
                    analogs = new_analogs

            if len(analogs) < required_analogs:
                logger.warning(f"Only found {len(analogs)} analogs after applying all distance thresholds.")
        except Exception as e:
            logger.exception("Error during geographic proximity search.")

    # Step 4: Finalize analogs
    if analogs.empty:
        logger.warning("No analogs found for the given entry.")
        return (np.nan, np.nan, np.nan, 0) + tuple([None] * required_analogs)

    # Ensure 'price_per_square_meter' has valid numeric values
    analogs = analogs.dropna(subset=["price_per_square_meter"])
    if analogs.empty:
        logger.warning("No analogs with valid 'price_per_square_meter' found.")
        return (np.nan, np.nan, np.nan, 0) + tuple([None] * required_analogs)

    # Compute statistics
    median_price = analogs["price_per_square_meter"].median()
    max_price = analogs["price_per_square_meter"].max()
    min_price = analogs["price_per_square_meter"].min()
    num_analogs = len(analogs)

    # Extract analog links, ensuring there are enough links
    links = analogs["link"].dropna().unique().tolist()
    analog_links = links[:required_analogs]
    while len(analog_links) < required_analogs:
        analog_links.append(None)  # Fill with None if not enough links

    logger.info(f"Returning statistics and analog links: Median={median_price}, Max={max_price}, "
                f"Min={min_price}, Count={num_analogs}, Links={analog_links}")

    return (median_price, max_price, min_price, num_analogs) + tuple(analog_links)


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

def get_flat_features(entry: pd.Series) -> pd.Series:
    """
    Returns a Series with the features of the flat.
    """
    logger.info("FEATURE EXTRACTION")
    logger.info(f"INITIAL ENTRY: {entry}")
    try:
        city = entry["city"].upper()
        district = entry["district"].upper()
        street = entry["street"].upper()
        house_number = entry["home_number"].upper()
    except AttributeError:
        city = None
        district = None
        street = None
        house_number = None

    housing_comlex_name = (
        entry["residential_complex"].upper()
        if entry["residential_complex"] is not None
        else ""
    )
    total_square_meters = entry["total_square"]
    wall_type = (
        entry["building_type"].upper()
        if entry["building_type"] != "НЕИЗВЕСТНЫЙ"
        else "ИНОЕ"
    )
    floor = entry["flat_floor"]
    floors_number = entry["building_floor"]
    rooms_number = entry["live_rooms"]
    construction_year = entry["building_year"]
    bathroom = entry["flat_toilet"].upper()
    former_hostel = True if entry["flat_priv_dorm"] == "Да" else False

    location = None
    if entry["latitude"] is not None and entry["longitude"] is not None:
        latitude = entry["latitude"]
        longitude = entry["longitude"]
        logger.info(f"Latitude and Longitude provided: {latitude}, {longitude}")
        # address
        location = Nominatim(user_agent="my_app").reverse(f"{latitude}, {longitude}")
        logger.info(f"Location Found: {location.address}")
    else:
        location = get_location(
            city, district, street, house_number, housing_comlex_name
        )
        if location is None:
            logger.error("Could not find location for the flat.")
            return None
        logger.info(f"Location Found: {location.address}")

    latitude = location.latitude
    longitude = location.longitude
    analog_prices_median = None
    analog_prices_max = None
    analog_price_min = None
    
    housing_comlex_name = "НЕТ" if housing_comlex_name == "" else housing_comlex_name

    entry = pd.Series(
        [
            latitude,
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
            analog_price_min,
        ],
        index=MODEL_COLUMNS,
    )

    logger.info(f"ENTRY: {entry}")

    logger.info("SEARCHING ANALOGS")
    (
        analog_prices_median,
        analog_prices_max,
        analog_prices_min,
        analogs_number,
        analog_link_1,
        analog_link_2,
        analog_link_3,
    ) = get_analog_prices_for_entry(analogs, entry)

    (
        entry["analog_prices_median"],
        entry["analog_prices_max"],
        entry["analog_prices_min"],
        entry["analogs_found"],
        entry["analog_1"],
        entry["analog_2"],
        entry["analog_3"],
    ) = get_analog_prices_for_entry(analogs, entry)
    
    if location:
        entry['address_geocoder'] = location.address
    else:
        entry['address_geocoder'] = "Не найдено"

    logger.info(f"ENTRY AFTER ANALOGS: {entry}")

    return entry
