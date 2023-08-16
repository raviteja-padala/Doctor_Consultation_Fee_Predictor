from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('DT_model.sav')

# List of Specialities and Zones
specialities = [
    'Dentist', 'Physiotherapist', 'General physician', 'Gynecologist', 'Psychiatrist',
    'Ayurveda', 'Dermatologist', 'Orthopedic', 'Pediatrician', 'Cardiology',
    'Homeopathy', 'ENT', 'Neurologist', 'Urologist', 'Gastroenterologist'
]

zones = ['East', 'West', 'North', 'South', 'Central']

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_fee = None

    if request.method == 'POST':
        speciality = request.form['speciality']
        zone = request.form['zone']
        years_of_experience = float(request.form['years_of_experience'])

        encoded_speciality = np.zeros(len(specialities))
        encoded_speciality[specialities.index(speciality)] = 1

        encoded_zone = np.zeros(len(zones))
        encoded_zone[zones.index(zone)] = 1

        feature_vector = np.hstack((encoded_speciality, encoded_zone, years_of_experience))

        predicted_fee = model.predict([feature_vector])[0]

    return render_template('index.html', specialities=specialities, zones=zones, predicted_fee=predicted_fee)

if __name__ == '__main__':
    app.run(debug=True)
