import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('DT_model.sav')

# List of Specialities and Zones
specialities = [
    'Dentist', 'Physiotherapist', 'General physician', 'Gynecologist', 'Psychiatrist',
    'Ayurveda', 'Dermatologist', 'Orthopedic', 'Pediatrician', 'Cardiology',
    'Homeopathy', 'ENT', 'Neurologist', 'Urologist', 'Gastroenterologist'
]

zones = ['East', 'West', 'North', 'South', 'Central']

# Streamlit UI
st.title("Consultation Fee Predictor")

st.sidebar.header("Input Features")

# User input: Speciality
speciality = st.sidebar.selectbox("Speciality", specialities)

# User input: Zone
zone = st.sidebar.selectbox("Zone", zones)

# User input: Years of Experience
years_of_experience = st.sidebar.number_input("Years of Experience", min_value=0, max_value=50, value=1)

# Submit button
submit_button = st.sidebar.button("Submit")

# Process upon clicking Submit
if submit_button:
    # Encode the Speciality and Zone
    encoded_speciality = np.zeros(len(specialities))
    encoded_speciality[specialities.index(speciality)] = 1

    encoded_zone = np.zeros(len(zones))
    encoded_zone[zones.index(zone)] = 1

    # Create a feature vector
    feature_vector = np.hstack((encoded_speciality, encoded_zone, years_of_experience))

    # Predict the Consultation Fee
    predicted_fee = model.predict([feature_vector])[0]

    # Display the predicted fee
    #st.subheader("Predicted Consultation Fee")
    #st.write(f"The predicted consultation fee is ${predicted_fee:.2f}")


    # Display the predicted fee
    st.subheader("Predicted Consultation Fee")
    st.write(f"Consultation fee of \"{speciality}\" in {zone} zone is ₹{predicted_fee:.2f}")

