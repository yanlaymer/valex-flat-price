import pandas as pd
import numpy as np
from loguru import logger
from geopy.geocoders import Nominatim, ArcGIS, GoogleV3
from scipy.spatial import KDTree
from itertools import combinations
from src.constants import MODEL_COLUMNS
from fuzzywuzzy import fuzz

analogs = pd.read_csv("data/analogs_50K.csv", compression='gzip') # db in rl

def get_analog_prices_for_entry(data: pd.DataFrame, entry: dict, required_analogs: int = 5) -> tuple:
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
                        "price_per_square_meter", "link", 'condition', 'wall_type']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logger.error(f"The data is missing required columns: {missing_columns}")
        raise ValueError(f"The data is missing required columns: {missing_columns}")

    # Validate required keys in entry
    required_keys = ["rooms_number", "construction_year", "housing_comlex_name", "latitude", "longitude", 'condition', 'wall_type']
    missing_keys = [key for key in required_keys if key not in entry]
    if missing_keys:
        logger.error(f"The entry is missing required keys: {missing_keys}")
        raise KeyError(f"The entry is missing required keys: {missing_keys}")

    try:
        rooms_number = entry["rooms_number"]
        wall_type = entry['wall_type'].upper()

        logger.info(f"WALL TYPE OF ENTRY: {wall_type}")
        # First, filter data to entries with the same rooms_number
        filtered_data = data[data["rooms_number"] == rooms_number].copy()
        if entry['condition'] == 'Черновая отделка':
            filtered_data = filtered_data[filtered_data.condition == entry['condition']]
        filtered_data = filtered_data[filtered_data['wall_type'].str.upper() == wall_type]
        logger.info(f"Filtered data with exact rooms_number={rooms_number}: {filtered_data.shape[0]} records found.")
    except Exception as e:
        logger.exception("Error filtering data based on rooms_number.")
        raise e

    # Initialize analogs DataFrame
    analogs = pd.DataFrame()

    # Step 2: Fuzzy match on housing_complex_name if applicable
    housing_complex_name = entry.get("housing_comlex_name", "").strip().upper()

    if housing_complex_name and housing_complex_name != "NONE":
        try:
            threshold = 85  # Define fuzzy matching threshold
            scores = filtered_data["housing_comlex_name"].fillna("").str.upper().apply(
                lambda x: fuzz.token_set_ratio(x, housing_complex_name)
            )
            filtered_data_with_scores = filtered_data.assign(score=scores)
            analogs = filtered_data_with_scores[filtered_data_with_scores['score'] >= threshold]
            analogs = analogs.sort_values(by='score', ascending=False)
            analogs['source'] = 'complex_name'  # Mark source
            logger.info(f"Found {len(analogs)} analogs by housing_complex_name with name '{housing_complex_name}'.")
        except Exception as e:
            logger.exception("Error during fuzzy matching of housing_complex_name.")

    # Step 3: If not enough analogs found, use geographic proximity
    if len(analogs) < required_analogs:
        try:
            # Define distance thresholds in kilometers
            distance_thresholds = [1, 3, 5, 10, 15, 20, 50]  # Adjust as needed
            entry_coords_rad = np.radians([entry["latitude"], entry["longitude"]])
            tree = KDTree(np.radians(filtered_data[["latitude", "longitude"]].values))

            # Initialize an empty DataFrame to accumulate analogs
            accumulated_analogs = analogs.copy()

            for threshold_km in distance_thresholds:
                distance_limit_rad = threshold_km / 6371.0088  # Earth's radius in kilometers

                # Query the tree for points within the distance limit
                indices = tree.query_ball_point(entry_coords_rad.reshape(1, -1), distance_limit_rad)
                indices = indices[0]  # query_ball_point returns a list of arrays

                # Filter valid indices
                valid_indices = [i for i in indices if i < len(filtered_data)]
                new_analogs = filtered_data.iloc[valid_indices]
                new_analogs = new_analogs.copy()
                new_analogs['source'] = 'proximity'  # Mark source

                # Combine with accumulated analogs
                accumulated_analogs = pd.concat([accumulated_analogs, new_analogs]).drop_duplicates()

                if len(accumulated_analogs) >= required_analogs:
                    logger.info(f"Found {len(accumulated_analogs)} analogs within {threshold_km} km.")
                    break
                else:
                    logger.info(f"Only found {len(accumulated_analogs)} analogs within {threshold_km} km. Expanding search...")

            if len(accumulated_analogs) < required_analogs:
                logger.warning(f"Only found {len(accumulated_analogs)} analogs after applying all distance thresholds.")

            analogs = accumulated_analogs

        except Exception as e:
            logger.exception("Error during geographic proximity search.")

    # Step 4: If not enough analogs found, expand rooms_number criteria
    if len(analogs) < required_analogs:
        logger.info("Not enough analogs found with exact rooms_number. Expanding rooms_number criteria.")
        try:
            # Include rooms_number -1 and +1
            mask_same_room = data["rooms_number"] == rooms_number
            mask_room_lower = data["rooms_number"] == rooms_number - 1
            mask_room_upper = data["rooms_number"] == rooms_number + 1
            expanded_data = data[mask_same_room | mask_room_lower | mask_room_upper].copy()
            if entry['condition'] == 'Черновая отделка':
                expanded_data = expanded_data[expanded_data.condition == entry['condition']]
            expanded_data = expanded_data[expanded_data['wall_type'].str.upper() == wall_type]
            logger.info(f"Expanded data with rooms_number +/-1: {expanded_data.shape[0]} records found.")

            # Re-run steps 2 and 3 with expanded data
            analogs = pd.DataFrame()

            # Step 2: Fuzzy match on housing_complex_name if applicable
            if housing_complex_name and housing_complex_name != "NONE":
                try:
                    scores = expanded_data["housing_comlex_name"].fillna("").str.upper().apply(
                        lambda x: fuzz.token_set_ratio(x, housing_complex_name)
                    )
                    expanded_data_with_scores = expanded_data.assign(score=scores)
                    analogs = expanded_data_with_scores[expanded_data_with_scores['score'] >= threshold]
                    analogs = analogs.sort_values(by='score', ascending=False)
                    analogs['source'] = 'complex_name'  # Mark source
                    logger.info(f"Found {len(analogs)} analogs by housing_complex_name with expanded rooms_number.")
                except Exception as e:
                    logger.exception("Error during fuzzy matching of housing_complex_name on expanded data.")

            # Step 3: Geographic proximity with expanded data
            if len(analogs) < required_analogs:
                try:
                    tree = KDTree(np.radians(expanded_data[["latitude", "longitude"]].values))
                    accumulated_analogs = analogs.copy()

                    for threshold_km in distance_thresholds:
                        distance_limit_rad = threshold_km / 6371.0088  # Earth's radius in kilometers

                        # Query the tree for points within the distance limit
                        indices = tree.query_ball_point(entry_coords_rad.reshape(1, -1), distance_limit_rad)
                        indices = indices[0]  # query_ball_point returns a list of arrays

                        # Filter valid indices
                        valid_indices = [i for i in indices if i < len(expanded_data)]
                        new_analogs = expanded_data.iloc[valid_indices]
                        new_analogs = new_analogs.copy()
                        new_analogs['source'] = 'proximity'  # Mark source

                        # Combine with accumulated analogs
                        accumulated_analogs = pd.concat([accumulated_analogs, new_analogs]).drop_duplicates()

                        if len(accumulated_analogs) >= required_analogs:
                            logger.info(f"Found {len(accumulated_analogs)} analogs within {threshold_km} km with expanded rooms_number.")
                            break
                        else:
                            logger.info(f"Only found {len(accumulated_analogs)} analogs within {threshold_km} km with expanded rooms_number. Expanding search...")

                    if len(accumulated_analogs) < required_analogs:
                        logger.warning(f"Only found {len(accumulated_analogs)} analogs after applying all distance thresholds on expanded data.")

                    analogs = accumulated_analogs

                except Exception as e:
                    logger.exception("Error during geographic proximity search on expanded data.")

        except Exception as e:
            logger.exception("Error expanding rooms_number criteria.")

    # Step 5: Apply IQR outlier drop if a lot of analogs are found
    if len(analogs) >= 10:
        try:
            # Apply IQR outlier removal
            Q1 = analogs['price_per_square_meter'].quantile(0.25)
            Q3 = analogs['price_per_square_meter'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            initial_count = len(analogs)
            analogs = analogs[(analogs['price_per_square_meter'] >= lower_bound) & (analogs['price_per_square_meter'] <= upper_bound)]
            filtered_count = len(analogs)
            logger.info(f"Applied IQR outlier removal. Reduced analogs from {initial_count} to {filtered_count}.")
        except Exception as e:
            logger.exception("Error during IQR outlier removal.")

    # Step 6: Finalize analogs
    if analogs.empty:
        logger.warning("No analogs found for the given entry.")
        return (np.nan, np.nan, np.nan, 0) + tuple([None] * 3)

    # Ensure 'price_per_square_meter' has valid numeric values
    analogs = analogs.dropna(subset=["price_per_square_meter"])
    if analogs.empty:
        logger.warning("No analogs with valid 'price_per_square_meter' found.")
        return (np.nan, np.nan, np.nan, 0) + tuple([None] * 3)

    # Compute statistics
    median_price = analogs["price_per_square_meter"].median()
    max_price = analogs["price_per_square_meter"].max()
    min_price = analogs["price_per_square_meter"].min()
    num_analogs = len(analogs)

    # Sort analogs so that those from complex name matching come first
    analogs['source'] = analogs['source'].fillna('proximity')  # Fill missing sources with 'proximity'
    analogs['source_order'] = analogs['source'].map({'complex_name': 0, 'proximity': 1})
    analogs = analogs.sort_values(by=['source_order', 'score'], ascending=[True, False])

    # Extract analog links, ensuring there are enough links
    links = analogs["link"].dropna().unique().tolist()
    analog_links = links[:required_analogs]
    while len(analog_links) < required_analogs:
        analog_links.append(None)  # Fill with None if not enough links

    logger.info(f"Returning statistics and analog links: Median={median_price}, Max={max_price}, "
                f"Min={min_price}, Count={num_analogs}, Links={analog_links}")

    return (median_price, max_price, min_price, num_analogs) + tuple(analog_links[:3])


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
    condition = entry['flat_renovation']

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
    
    entry['condition'] = condition

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
