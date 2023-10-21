import streamlit as st
from src.predictor import Predictor
import hashlib
from pickle import load
import pandas as pd
from time import sleep

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
st.write("Заполните следующие поля, чтобы получить оценку стоимости вашей квартиры.")

def main():
    # Location Information
    st.subheader("Информация о местоположении 📍")
    st.markdown(
    """
    <div style="color: orange; font-size: small; margin-bottom: 20px">
    ⚠️ Поиск по адресу может быть некорректным. Рекомендуется использовать широту и долготу.
    </div>
    """,
    unsafe_allow_html=True,
)
    input_type = st.radio("Как вы хотите указать местоположение?", ["Введите адрес", "Введите широту и долготу"])
    
    if input_type == "Введите адрес":
        city = st.text_input("Город")
        district = st.text_input("Район")
        street = st.text_input("Улица")
        home_number = st.text_input("Номер дома")
        latitude = None
        longitude = None
    else:
        city = None
        district = None
        street = None
        home_number = None
        latitude = st.text_input("Широта (формат: xx.xxxxxx)")
        longitude = st.text_input("Долгота (формат: xx.xxxxxx)")
        
        try:
            latitude = float(latitude)
            longitude = float(longitude)
        except ValueError:
            st.warning("Пожалуйста, введите корректные значения для широты и долготы.")

    # Apartment Details
    st.subheader("Детали квартиры 🏠")
    residential_complex = st.text_input("Жилой комплекс (если есть)", value="")
    total_square = st.number_input("Общая площадь (в кв.м)", min_value=0.0)
    kitchen_square = st.number_input("Площадь кухни (в кв.м)", min_value=0.0)
    flat_floor = st.number_input("Этаж квартиры", min_value=1)
    building_floor = st.number_input("Этаж здания", min_value=1)
    live_rooms = st.number_input("Количество жилых комнат", min_value=1)
    building_year = st.number_input("Год постройки", min_value=1900, max_value=2023)

    building_type_choices = encoders["building"].classes_.tolist()
    building_type = st.selectbox("Тип здания", building_type_choices)

    flat_priv_dorm_choices = encoders["flat_priv"].classes_.tolist()
    flat_priv_dorm = st.selectbox("Частное общежитие", flat_priv_dorm_choices)
    flat_renovation_choices = encoders["flat_renovation"].classes_.tolist()
    flat_renovation = st.selectbox("Ремонт в квартире", flat_renovation_choices)
    flat_toilet_choices = encoders["toilet"].classes_.tolist()
    flat_toilet = st.selectbox("Туалет", flat_toilet_choices)
    live_furniture_choices = encoders["furniture"].classes_.tolist()
    live_furniture = st.selectbox("Мебель", live_furniture_choices)

    # Button to start prediction
    if st.button("Оценить стоимость квартиры"):
        
        data = {
            "city": city,
            "district": district,
            "street": street,
            "residential_complex": residential_complex,
            "latitude": latitude,
            "longitude": longitude,
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
        status_placeholder = st.empty()
        status_placeholder.text("Находим признаки... ⏳")
        sleep(1)  # Simulating some processing time
        predictor = Predictor(data)
        price, links = predictor.predict_price()
        
        status_placeholder.text("Сверяемся с аналогами... ⏳")
        sleep(0.5)  # Simulating some processing time

        # Update the status and show a loader icon

        status_placeholder.text("Получение оценки... ⏳")
        sleep(1)  # Simulating some processing time
        st.success(f"Оценка квартиры: {price} Т")
        status_placeholder.text("Готово ✅")

        # plot map
        lat, lon = predictor.model_entry['latitude'], predictor.model_entry['longitude']
        map_df = pd.DataFrame({'lat': [lat], 'lon': [lon]})
        st.map(map_df, zoom=15)

        # make preview links as buttons
        st.write("Ссылки на аналоги:")
        for link in links:
            st.markdown(f'<a href="{link}" target="_blank">Ссылка</a>', unsafe_allow_html=True)
            
        st.write("Распределение цен")
        analog_prices_min = predictor.model_entry['analog_prices_min'] * predictor.entry['total_square']
        analog_prices_median = predictor.model_entry['analog_prices_median'] * predictor.entry['total_square']
        analog_prices_max = predictor.model_entry['analog_prices_max'] * predictor.entry['total_square']
        st.write(f"Минимальная цена: {analog_prices_min}")
        st.write(f"Медианная цена: {analog_prices_median}")
        st.write(f"Максимальная цена: {analog_prices_max}")
            

def auth():
    username = st.sidebar.text_input("Логин")
    password = st.sidebar.text_input("Пароль", type="password")
    if st.sidebar.button("Войти"):
        stored_password = hashlib.sha256("KNAGT2001_d".encode()).hexdigest()
        if username == "test" and check_password(stored_password, password):
            st.session_state.logged_in = True
        else:
            st.warning("Некорректные данные. Попробовать еще раз.")
    return st.session_state.logged_in

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if auth():
    main()
else:
    st.sidebar.info("Пожалуйста, введите данные для авторизаций.")
