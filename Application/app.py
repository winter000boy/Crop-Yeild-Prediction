from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle

# while training the model whatever the scikit learn version we have used we have to use the same version here also 1.5.2
# pip install scikit-learn==1.5.2
import sklearn

CropYeild = pickle.load(open('CropYeild.pkl', 'rb'))
Preprocessor = pickle.load(open('Preprocessor.pkl', 'rb'))

# Creating flask main
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Year = request.form['year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']
        Area = request.form['Area']
        Item = request.form['Item']

        features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]])

        # Now we have to one-hot encode and standardize this input also
        transformed_features = Preprocessor.transform(features)
        predicted_value = CropYeild.predict(transformed_features).reshape(1, -1)
        return render_template('index.html', predicted_value='Predicted Crop Yield is {}'.format(predicted_value))

if __name__ == '__main__':
    app.run(debug=True)