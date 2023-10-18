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

st.title("VALEX ОЦЕНКА КВАРТИРЫ")

# Input fields for the FlatModel
def main():
    city = st.text_input("Город")
    
    district_choices = encoders["district"].classes_.tolist()
    district = st.selectbox("Район", district_choices)
    
    street = st.text_input("Улица")
    residential_complex = st.text_input("Жилой комплекс (необязательно)", value="")
    home_number = st.text_input("Номер дома")
    
    building_type_choices = encoders["building"].classes_.tolist()
    building_type = st.selectbox("Тип здания", building_type_choices)
    
    total_square = st.number_input("Общая площадь (в кв.м)", min_value=0.0)
    kitchen_square = st.number_input("Площадь кухни (в кв.м)", min_value=0.0)
    flat_floor = st.number_input("Этаж квартиры", min_value=1)
    building_floor = st.number_input("Этаж здания", min_value=1)
    live_rooms = st.number_input("Количество жилых комнат", min_value=1)
    building_year = st.number_input("Год постройки", min_value=1900, max_value=2023)
    
    flat_priv_dorm_choices = encoders["flat_priv"].classes_.tolist()
    flat_priv_dorm = st.selectbox("Частное общежитие", flat_priv_dorm_choices)
    
    flat_renovation_choices = encoders["flat_renovation"].classes_.tolist()
    flat_renovation = st.selectbox("Ремонт в квартире", flat_renovation_choices)
    
    flat_toilet_choices = encoders["toilet"].classes_.tolist()
    flat_toilet = st.selectbox("Туалет", flat_toilet_choices)
    
    live_furniture_choices = encoders["furniture"].classes_.tolist()
    live_furniture = st.selectbox("Мебель", live_furniture_choices)

    if st.button("Оценить стоимость квартиры"):
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
            st.success(f"{result.get('status_code')} : Оценочная стоимость: {round(result.get('price'), -4)}, message: {result.get('message')}")
        else:
            st.error(f"{result.get('status_code')} : {result.get('message')}")
            

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        stored_password = hashlib.sha256("KNAGT2001_d".encode()).hexdigest()
        if username == "test" and check_password(stored_password, password):
            st.session_state.logged_in = True
        else:
            st.warning("Некорректные данные. Попробовать еще раз.")

if st.session_state.logged_in:
    main()
else:
    st.info("Пожалуйста, введите данные для авторизаций.")
