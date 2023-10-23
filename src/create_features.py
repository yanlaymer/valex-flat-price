import pandas as pd
import numpy as np
from loguru import logger
from geopy.geocoders import Nominatim, ArcGIS
from scipy.spatial import KDTree
from itertools import combinations
from fuzzywuzzy import fuzz
from src.constants import MODEL_COLUMNS

analogs = pd.read_csv("data/analogs.csv", compression="gzip")


def get_analog_prices_for_entry(data, entry):
    mask = (data["rooms_number"] == entry["rooms_number"]) & (
        np.abs(data["construction_year"] - entry["construction_year"]) <= 7
    )
    filtered_data = data[mask]

    analogs = None
    if entry["housing_comlex_name"] not in ["None", ""]:
        entry["housing_comlex_name"] = entry["housing_comlex_name"].upper()
        threshold = 85
        mask = filtered_data["housing_comlex_name"].apply(
            lambda x: fuzz.ratio(x, entry["housing_comlex_name"]) >= threshold
        )
        analogs = filtered_data[mask]

    if analogs.empty or len(analogs) < 5:
        tree = KDTree(np.radians(filtered_data[["latitude", "longitude"]].values))
        distance_limit_rad = 1 / 6371.0088
        _, indices = tree.query(
            [np.radians(entry["latitude"]), np.radians(entry["longitude"])],
            distance_upper_bound=distance_limit_rad,
            k=3,
        )
        indices = np.atleast_1d(indices)
        analogs = filtered_data.iloc[indices]

        if analogs.empty or len(analogs) < 3:
            distance_limit_rad = 3 / 6371.0088
            _, indices = tree.query(
                [np.radians(entry["latitude"]), np.radians(entry["longitude"])],
                distance_upper_bound=distance_limit_rad,
                k=5,
            )
            analogs = filtered_data.iloc[indices]

    return (
        analogs["price_per_square_meter"].median(),
        analogs["price_per_square_meter"].max(),
        analogs["price_per_square_meter"].min(),
        len(analogs),
        analogs["link"].tolist()[0],
        analogs["link"].tolist()[1],
        analogs["link"].tolist()[2],
    )


def get_location(city, district, street, house_number, housing_comlex_name):
    geolocator = Nominatim(user_agent="another_app")
    geolocator_v2 = ArcGIS(user_agent="fallback_app")

    components = {
        "city": city.upper(),
        "district": district.upper()
        .replace("МКР", "МИКРОРАЙОН")
        .replace("Р-Н", "РАЙОН")
        + (" РАЙОН" if "РАЙОН" not in district else ""),
        "street": street.upper().replace("МКР", "МИКРОРАЙОН"),
        "house_number": house_number.upper(),
        "housing_comlex_name": housing_comlex_name.upper()
        if housing_comlex_name
        else "НЕТ",
    }

    all_combinations = [
        ", ".join(item[1] for item in subset)
        for r in range(len(components), 0, -1)
        for subset in combinations(
            sorted(components.items(), key=lambda x: -len(x[1])), r
        )
    ]

    for address in all_combinations:
        try:
            location = geolocator.geocode(address)
            if location:
                return location
        except:
            location = geolocator_v2.geocode(address)
            if location:
                return location

    return None


def get_flat_features(entry: pd.Series) -> pd.Series:
    try:
        city, district, street, house_number = (
            entry["city"].upper(),
            entry["district"].upper(),
            entry["street"].upper(),
            entry["home_number"].upper(),
        )
    except AttributeError:
        city, district, street, house_number = None, None, None, None

    housing_comlex_name = (
        entry["residential_complex"].upper() if entry["residential_complex"] else "НЕТ"
    )

    if entry.get("latitude") and entry.get("longitude"):
        location = Nominatim(user_agent="my_app").reverse(
            f"{entry['latitude']}, {entry['longitude']}"
        )
    else:
        location = get_location(
            city, district, street, house_number, housing_comlex_name
        )
        if not location:
            return None

    entry_data = [
        location.latitude,
        location.longitude,
        entry["flat_floor"],
        entry["building_floor"],
        entry["live_rooms"],
        entry["total_square"],
        entry["building_year"],
        entry["building_type"].upper()
        if entry["building_type"] != "НЕИЗВЕСТНЫЙ"
        else "ИНОЕ",
        housing_comlex_name,
        entry["flat_toilet"].upper(),
        True if entry["flat_priv_dorm"] == "Да" else False,
        None,
        None,
        None,
    ]

    entry = pd.Series(entry_data, index=MODEL_COLUMNS)
    (
        entry["analog_prices_median"],
        entry["analog_prices_max"],
        entry["analog_prices_min"],
        entry["analogs_found"],
        entry["analog_1"],
        entry["analog_2"],
        entry["analog_3"],
    ) = get_analog_prices_for_entry(analogs, entry)

    return entry
