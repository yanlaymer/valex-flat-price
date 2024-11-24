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

    # Step 1: Filter based on exact rooms match
    try:
        rooms_number = entry["rooms"]
        filtered_data = data[data["rooms"] == rooms_number].copy()
        logger.info(f"Filtered data based on rooms={rooms_number}: {filtered_data.shape[0]} records found.")
        if not filtered_data.empty:
            feedback["rooms"] = True
    except Exception as e:
        logger.exception("Error filtering data based on rooms.")
        raise e

    # Step 2: Filter based on building_type (wall_material)
    try:
        building_type = entry["building_type"]
        filtered_data = filtered_data[filtered_data["building_type"] == building_type]
        logger.info(f"Filtered data based on building_type={building_type}: {filtered_data.shape[0]} records found.")
        if not filtered_data.empty:
            feedback["building_type"] = True
    except Exception as e:
        logger.exception("Error filtering data based on building_type.")
        raise e

    # Step 3: Filter based on year_built within ±5 years
    try:
        year_built = entry["year_built"]
        year_min = year_built - 5
        year_max = year_built + 5
        filtered_data = filtered_data[
            (filtered_data["year_built"] >= year_min) &
            (filtered_data["year_built"] <= year_max)
        ]
        logger.info(f"Filtered data based on year_built between {year_min}-{year_max}: {filtered_data.shape[0]} records found.")
        if not filtered_data.empty:
            feedback["year_built"] = True
    except Exception as e:
        logger.exception("Error filtering data based on year_built.")
        raise e

    # Step 4: Filter based on total_floors
    try:
        total_floors = entry["total_floors"]
        filtered_data = filtered_data[filtered_data["total_floors"] == total_floors]
        logger.info(f"Filtered data based on total_floors={total_floors}: {filtered_data.shape[0]} records found.")
        if not filtered_data.empty:
            feedback["total_floors"] = True
        else:
            # Relax criterion if building height exceeds 9 floors
            if total_floors > 9:
                filtered_data = data[data["total_floors"] > 9]
                logger.info(f"Relaxed total_floors criterion for buildings over 9 floors: {filtered_data.shape[0]} records found.")
    except Exception as e:
        logger.exception("Error filtering data based on total_floors.")
        raise e

    # Step 5: Filter based on floor
    try:
        flat_floor = entry["floor"]
        total_floors = entry["total_floors"]
        if flat_floor == 1 or flat_floor == total_floors:
            # First or last floor
            filtered_data = filtered_data[
                (filtered_data["floor"] == 1) |
                (filtered_data["floor"] == filtered_data["total_floors"])
            ]
        else:
            # Middle floors
            filtered_data = filtered_data[
                (filtered_data["floor"] != 1) &
                (filtered_data["floor"] != filtered_data["total_floors"])
            ]
        logger.info(f"Filtered data based on floor: {filtered_data.shape[0]} records found.")
        if not filtered_data.empty:
            feedback["floor"] = True
    except Exception as e:
        logger.exception("Error filtering data based on floor.")
        raise e

    # Step 6: Filter based on square (area within 15%–20%)
    try:
        total_square = entry["square"]
        area_min = total_square * 0.8  # 20% less
        area_max = total_square * 1.15  # 15% more
        filtered_data = filtered_data[
            (filtered_data["square"] >= area_min) &
            (filtered_data["square"] <= area_max)
        ]
        logger.info(f"Filtered data based on square between {area_min}-{area_max}: {filtered_data.shape[0]} records found.")
        if not filtered_data.empty:
            feedback["square"] = True
    except Exception as e:
        logger.exception("Error filtering data based on square.")
        raise e

    # Step 7: Filter based on location
    try:
        city = entry["город"].upper()
        entry_coords_rad = np.radians([entry["latitude"], entry["longitude"]])
        if city in ["АЛМАТЫ", "ШЫМКЕНТ", "АСТАНА"]:
            distance_thresholds = [2, 3, 5, 10]  # Start with 2 km
        else:
            distance_thresholds = [5, 10, 15, 20]  # Start with 5 km

        # Build KDTree
        tree = KDTree(np.radians(filtered_data[["latitude", "longitude"]].values))

        analogs = pd.DataFrame()

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
            analogs = pd.concat([analogs, new_analogs]).drop_duplicates()

            if len(analogs) >= 10:
                logger.info(f"Found {len(analogs)} analogs within {threshold_km} km.")
                feedback["location"] = True
                break
            else:
                logger.info(f"Only found {len(analogs)} analogs within {threshold_km} km. Expanding search...")

        if analogs.empty:
            logger.warning("No analogs found based on location.")
    except Exception as e:
        logger.exception("Error during geographic proximity search.")
        raise e

    # Step 8: Fuzzy match on residential_complex_name if applicable
    try:
        residential_complex_name = entry.get("residential_complex_name", "").strip().upper()
        if residential_complex_name and residential_complex_name != "НЕТ":
            threshold = 85  # Define fuzzy matching threshold
            scores = analogs["residential_complex_name"].fillna("").str.upper().apply(
                lambda x: fuzz.token_set_ratio(x, residential_complex_name)
            )
            analogs = analogs.assign(score=scores)
            analogs = analogs[analogs['score'] >= threshold]
            analogs = analogs.sort_values(by='score', ascending=False)
            analogs['source'] = 'complex_name'  # Mark source
            logger.info(f"Found {len(analogs)} analogs by residential_complex_name with name '{residential_complex_name}'.")
    except Exception as e:
        logger.exception("Error during fuzzy matching of residential_complex_name.")

    # Ensure at least 10 analogs
    if len(analogs) < 10:
        logger.warning(f"Only found {len(analogs)} analogs after applying all filters. Applying fallback logic.")

        # Fallback logic: progressively relax filters
        # Example: Relax area filtering
        if not feedback["square"]:
            area_min = total_square * 0.7  # 30% less
            area_max = total_square * 1.3  # 30% more
            filtered_data = data[
                (data["square"] >= area_min) &
                (data["square"] <= area_max)
            ]
            feedback["square"] = False  # Parameter was relaxed
            logger.info(f"Relaxed area filtering to {area_min}-{area_max}. Records found: {filtered_data.shape[0]}")

        # Re-apply location filtering with relaxed area
        # Additional fallback logic can be implemented as needed

    # Step 9: Finalize analogs
    if analogs.empty:
        logger.warning("No analogs found for the given entry after fallback.")
        feedback_status = "RED"
        return (np.nan, np.nan, np.nan, 0, feedback_status, feedback) + tuple([None] * required_analogs)

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
