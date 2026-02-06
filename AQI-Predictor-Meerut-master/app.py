import os
import pickle
import streamlit as st

import pandas as pd
import numpy as np

import requests
st.set_page_config(layout="wide", page_title="AQI Predictor")
import warnings
warnings.filterwarnings("ignore")

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'AQI', 'random_forest_regression_model.pkl')
    print(f"Loading model from: {model_path}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

def predict_aqi(features):
    prediction = model.predict(features)
    return prediction

def fetch_weather_data():
    url = "https://the-weather-api.p.rapidapi.com/api/weather/meerut"
    headers = {
        "x-rapidapi-key": "c667cc1068msh6202387f1d5addbp140cb2jsn678621e3d257",
        "x-rapidapi-host": "the-weather-api.p.rapidapi.com"
    }

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data and 'data' in data:
                weather_data = data['data']
                return weather_data
            else:
                return None
        else:
            return None
    except requests.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def home():
    st.title("AQI Predictor for Meerut City")
    description_col, image_col = st.columns([2, 1])
    with description_col:
        st.write("Meerut's Air Quality Index (AQI) has witnessed notable fluctuations over the past six years, reflecting the dynamic interplay of industrial activities, vehicular emissions, and seasonal variations. Despite occasional improvements spurred by regulatory measures and public awareness initiatives, AQI levels have predominantly lingered in the moderate to poor range, highlighting persistent challenges in maintaining air quality standards. To mitigate these levels, concerted efforts are imperative, including promoting sustainable practices, enforcing environmental regulations, and raising public awareness about the detrimental effects of air pollution. Additionally, during periods of heightened AQI, individuals should take precautions such as limiting outdoor exposure, using indoor air purifiers, and wearing protective masks to safeguard their health. By addressing these concerns collectively, we can strive towards a cleaner and healthier environment for Meerut.")
        st.write("Understanding and monitoring the AQI in Meerut is essential for improving public health and ensuring a better quality of life. Continuous efforts from both governmental and non-governmental entities are required to address the underlying causes of air pollution and implement effective mitigation strategies. Residents should stay informed and take proactive steps to reduce exposure to polluted air, especially during high AQI periods.")
    with image_col:
        st.image('Images/40059.jpg', caption="Air Quality Index (AQI) for Meerut", width=421, use_column_width=True)

    st.title("Real-time Weather Data for Meerut")
    st.write("Fetching real-time weather data...")
    weather_data = fetch_weather_data()
    if weather_data:
        aqi = weather_data.get('aqi')
        current_weather = weather_data.get('current_weather')
        temp = weather_data.get('temp')
        aqi_remark = weather_data.get('aqi_remark')
        st.write("AQI (Air Quality Index):", aqi)
        st.write("Current Weather Condition:", current_weather)
        st.write("Temperature:", temp, "°C")
        st.write("AQI Remark:", aqi_remark)
    else:
        st.error("Error fetching weather data. Please try again later.")

def about_us():
    st.title("About Us")
    st.write("We are final-year students of the B.Tech CSE (Data Science) program at Meerut Institute of Engineering and Technology, Meerut. Our passion for technology and environmental science has driven us to develop an AQI Predictor specifically for our city, Meerut.")
    st.write("In this project, we meticulously collected and analyzed air quality data from the Tutiempo website. By applying supervised machine learning algorithms, we have successfully created a model that predicts the Air Quality Index (AQI) for Meerut with an accuracy of 85%. Our model is built on the foundation of six years of historical data, and through rigorous testing and optimization, we have employed various machine learning techniques to enhance our predictions")
    st.write("Our project is not just a culmination of our academic journey but a step towards contributing to our community by providing real-time air quality insights. We believe that understanding and predicting air quality is crucial for the health and well-being of our residents.")
    st.write("Looking ahead, our goal is to integrate unsupervised machine learning algorithms to further improve the accuracy and reliability of our predictions. We are committed to continuous learning and innovation, and we aspire to set a benchmark in environmental data science.")
    st.title("Meet Our Team")
    team_members = [
        {"name": "Ayush Sharma", "position": "Team Leader - Backend Developer", "contact": "+91 9084626104", "image": "Images/ayush.jpg"},
        {"name": "Dhariya Rajput", "position": "Team Member - Content Writer", "contact": "+91 9760666142", "image": "Images/dd.jpg"},
        {"name": "Ankit .", "position": "Team Member - Content Writer", "contact": "+91 8126810022", "image": "Images/ankitimg.jpg"},
        {"name": "Sandeep Kumar", "position": "Team Member - UI Developer", "contact": "+91 7906774535", "image": "Images/sandeep2.jpg"}
    ]
    cols = st.columns(4)
    for idx, member in enumerate(team_members):
        with cols[idx]:
            st.image(member["image"], use_column_width=True, width=150)
            st.write(f"**{member['name']}**")
            st.write(f"*{member['position']}*")
            st.write(f"Contact: {member['contact']}")

def calculate():
    st.title("Calculate AQI of Meerut")
    st.write("Enter the Values")
    st.header('Input Features')
    temperature = st.slider('Average Temperature (°C)', 0.0, 40.0, 25.0)
    max_temperature = st.slider('Maximum Temperature (°C)', 0.0, 50.0, 30.0)
    min_temperature = st.slider('Minimum Temperature (°C)', 0.0, 30.0, 20.0)
    pressure = st.slider('Atmospheric Pressure (hPa)', 900, 1100, 1000)
    humidity = st.slider('Average Relative Humidity (%)', 0.0, 100.0, 50.0)
    visibility = st.slider('Average Visibility (Km)', 0.0, 20.0, 10.0)
    wind_speed = st.slider('Average Wind Speed (Km/h)', 0.0, 50.0, 10.0)
    pm25 = st.slider('PM2.5 (µg/m³)', 0.0, 500.0, 50.0)
    features = {'T': temperature, 'TM': max_temperature, 'Tm': min_temperature, 'SLP': pressure, 'H': humidity, 'VV': visibility, 'V': wind_speed, 'PM2.5': pm25}
    input_df = pd.DataFrame([features])
    if st.button('Predict AQI'):
        prediction = predict_aqi(input_df)
        st.subheader('Predicted AQI')
        st.write(prediction)
    st.subheader('Input Features')
    st.write(input_df)

def contact_us():
    st.title("Contact Us")
    st.write("Please fill out the form below to contact us.")
    with st.form("contact_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        message = st.text_area("Message")
        submitted = st.form_submit_button("Submit")
        if submitted:
            form_data = {"Name": name, "Email": email, "Message": message}
            response = requests.post("https://formspree.io/f/moqgznbd", data=form_data)
            if response.ok:
                st.success("Your message has been successfully submitted!")
            else:
                st.error("Failed to submit the form. Please try again later.")

pages = {"Home": home, "About Us": about_us, "Calculate": calculate, "Contact Us": contact_us}

def render_navbar():
    st.markdown("""
    <style>
    .navbar {display: flex; justify-content: space-between; align-items: center; background-color: #f8f9fa; padding: 10px 20px; margin-left: 25px;}
    .navbar img {height: 500px;}
    .navbar a {padding: 14px 20px; text-decoration: none; color: #000; font-size: 25px;}
    .navbar a:hover {background-color: #ddd; color: black;}
    </style>
    <div class="navbar">
        <div>
            <a href="/?page=Home">Home</a>
            <a href="/?page=About%20Us">About Us</a>
            <a href="/?page=Calculate">Calculate</a>
            <a href="/?page=Contact%20Us">Contact Us</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

params = st.experimental_get_query_params()
page = params.get("page", ["Home"])[0]
render_navbar()

if page in pages:
    pages[page]()
else:
    home()
