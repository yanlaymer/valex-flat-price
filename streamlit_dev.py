import streamlit as st
from src.predict import get_flat_price
import hashlib
from pickle import load


ARTEFACTS_PATH = 'model/artefacts/'
ENCODERS = {
    "district": "district_encoder.sav",
    "owner": "owner_encoder.sav",
    "building": "building_encoder.sav",
    "flat_renovation": "flat_renovation_encoder.sav",
    "flat_priv": "flat_priv_encoder.sav",
    "furniture": "furniture_encoder.sav",
    "toilet": "toilet_encoder.sav"
}

encoders = {}

for key, filename in ENCODERS.items():
    with open(ARTEFACTS_PATH + filename, 'rb') as f:
        encoders[key] = load(f)

def check_password(stored_password, user_password):
    return stored_password == hashlib.sha256(user_password.encode()).hexdigest()

# BASE_URL = "http://127.0.0.1:8002"  # Replace with the URL of your FastAPI server

# def get_flat_price(data):
#     response = requests.post(f"{BASE_URL}/get_flat_price/", json=data)
#     if response.status_code == 200:
#         return response.json()
#     else:
#         st.error("Failed to fetch the valuation.")
#         return None

st.title("Flat Price Estimation")

# Input fields for the FlatModel
def main():
    city = st.text_input("City")
    
    # Replace text inputs with dropdowns for the encoded categories
    district_choices = encoders["district"].classes_.tolist()
    district = st.selectbox("District", district_choices)
    
    street = st.text_input("Street")
    residential_complex = st.text_input("Residential Complex (Optional)", value="")
    home_number = st.text_input("Home Number")
    
    building_type_choices = encoders["building"].classes_.tolist()
    building_type = st.selectbox("Building Type", building_type_choices)
    
    total_square = st.number_input("Total Square (in sq.m)", min_value=0.0)
    kitchen_square = st.number_input("Kitchen Square (in sq.m)", min_value=0.0)
    flat_floor = st.number_input("Flat Floor", min_value=1)
    building_floor = st.number_input("Building Floor", min_value=1)
    live_rooms = st.number_input("Number of Living Rooms", min_value=1)
    building_year = st.number_input("Building Year", min_value=1900, max_value=2023)
    
    flat_priv_dorm_choices = encoders["flat_priv"].classes_.tolist()
    flat_priv_dorm = st.selectbox("Flat Priv Dorm", flat_priv_dorm_choices)
    
    flat_renovation_choices = encoders["flat_renovation"].classes_.tolist()
    flat_renovation = st.selectbox("Flat Renovation", flat_renovation_choices)
    
    flat_toilet_choices = encoders["toilet"].classes_.tolist()
    flat_toilet = st.selectbox("Flat Toilet", flat_toilet_choices)
    
    live_furniture_choices = encoders["furniture"].classes_.tolist()
    live_furniture = st.selectbox("Live Furniture", live_furniture_choices)

    if st.button("Estimate Flat Price"):
        data = {
            "city": city,
            "district": district,
            "street": street,
            "residential_complex": residential_complex,
            "home_number": home_number,
            "building_type": building_type,
            "total_square": total_square,
            "kitchen_square": kitchen_square,
            "flat_floor": flat_floor,
            "building_floor": building_floor,
            "live_rooms": live_rooms,
            "building_year": building_year,
            "flat_priv_dorm": flat_priv_dorm,
            "flat_renovation": flat_renovation,
            "flat_toilet": flat_toilet,
            "live_furniture": live_furniture
        }
        result = get_flat_price(data)
        if result.get("status_code") == 200:
            st.success(f"{result.get('status_code')} : Estimated Flat Price: {round(result.get('price'), -4)}, message: {result.get('message')}")
        else:
            st.error(f"{result.get('status_code')} : {result.get('message')}")
            
# Check for the session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# If not logged in, show the login form
if not st.session_state.logged_in:
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        # Password should be hashed and stored securely in the real world
        # For the sake of this example, we use a hardcoded username and password
        # Username: admin
        # Password: secret
        stored_password = hashlib.sha256("KNAGT2001_d".encode()).hexdigest()
        if username == "test" and check_password(stored_password, password):
            st.session_state.logged_in = True
        else:
            st.warning("Invalid credentials. Try again.")

if st.session_state.logged_in:
    main()  # Call your main app function
else:
    st.info("Please login to access this app.")
