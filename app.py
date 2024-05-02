from flask import Flask, jsonify, render_template, request
import io
import base64
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import urllib.parse
import pickle
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from datetime import timedelta

app = Flask(__name__)
train_test_data = {}

class LinearModel():
    def __init__(self, model=None):
        self.model = model

    def load_model(self, city):
        if city in train_test_data:
            self.model = train_test_data[city]['model']
        else:
            raise ValueError(f"Model for {city} not found.")

    def predict(self, city, X_test):
        if self.model is None:
            self.load_model(city)
        X_test_scaled = train_test_data[city]['scaler'].transform(X_test)
        predictions = self.model.predict(X_test_scaled)
        return predictions

def manage_dataset():
    # Read Datasets
    heat_index_data = pd.read_csv('./city_heat_index.csv')
    weather_data = pd.read_csv('./city_weather_data.csv')

    # Standardize Datasets
    weather_data['datetime'] = pd.to_datetime(weather_data['datetime'])
    weather_data['datetime'] = pd.to_datetime(weather_data['datetime'].dt.strftime('%Y-%m-%d'))
    heat_index_data['date_str'] = heat_index_data['year'].astype(str) + '-' + \
                                    heat_index_data['month'].astype(str).str.zfill(2) + '-' + \
                                    heat_index_data['day'].astype(str).str.zfill(2)
    heat_index_data['datetime'] = pd.to_datetime(heat_index_data['date_str'], format='%Y-%m-%d')
    heat_index_data = heat_index_data.drop('date_str', axis=1)

    # Merge datasets
    merged_data = pd.merge(weather_data, heat_index_data, on=['city_name', 'datetime'], how='inner')
    columns_to_drop = ['sys.type', 'sys.id', 'rain.1h', 'extraction_date_time', 'weather.icon', 'year']

    # Drop the columns
    merged_data = merged_data.drop(columns=columns_to_drop)
    merged_data['day_of_week'] = merged_data['datetime'].dt.dayofweek

    # Handle Invalid Data
    merged_data.interpolate(method='linear', inplace=True)

    merged_data['Month'] = merged_data["datetime"].dt.month

    # Apply linear interpolation to fill missing values (-999)
    merged_data['avg_heat_index_celsius'].replace(-999, np.nan, inplace=True)
    merged_data['avg_heat_index_celsius'].interpolate(method='linear', inplace=True)

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    features = ['main.temp', 'main.feels_like', 'main.temp_min', 'main.temp_max', 'main.pressure', 'wind.speed', 'wind.gust', 'main.humidity', 'main.sea_level', 'main.grnd_level', 'month', 'day', 'weather.id', 'coord.lat', 'coord.lon']
    merged_data[features] = imputer.fit_transform(merged_data[features])

    for city in merged_data['city_name'].unique():
        city_data = merged_data[merged_data['city_name'] == city]
        city_target = city_data['avg_heat_index_celsius']
        city_data.loc[:, features] = city_data[features].ffill()

        # Perform train-test split
        X_train, X_test, y_train, y_test = train_test_split(city_data[features], city_target, test_size=0.2, random_state=42)

        # Standardize the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        y_train_scaled = y_train.copy()

        # Train the model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train_scaled)
        print(f"{city} Model Score: {model.score(scaler.transform(X_test), y_test)}")

        # Store train-test split data and model in the dictionary
        train_test_data[city] = {
            'X_train': X_train_scaled,
            'X_test': X_test,
            'y_train': y_train_scaled,
            'y_test': y_test,
            'model': model,
            'scaler': scaler
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/select_city')
def select_city():
    return render_template('select_city.html')

@app.route('/city_list')
def city_list():
    cities = ['Baguio', 'Borongan', 'Butuan', 'Calapan', 'Catbalogan',
              'City of Masbate', 'Cotabato', 'Dagupan', 'Davao', 'Dipolog',
              'Dumaguete', 'General Santos', 'Laoag', 'Maasin', 'Malaybalay',
              'Puerto Princesa City', 'Quezon City', 'Roxas', 'San Jose',
              'Surigao City', 'Tacloban City', 'Tayabas', 'Tuguegarao',
              'Zamboanga City']
    return jsonify(cities)

@app.route('/process_city', methods=['POST'])
def process_city():
    data = request.json
    city = data['city']
    linear_model = LinearModel()
    predictions = linear_model.predict(city, train_test_data[city]['X_test'])
    return jsonify({'predictions': predictions.tolist()})


@app.route('/process_select_date', methods=['POST'])
def process_select_date():
    data = request.json
    date = data['date']
    
    print(date)
    return jsonify({ 'message': 'date retrieved', 'date': date })

@app.route('/select_date')
def select_date():
    city = request.args.get('city')
    img_str = request.args.get('img_str')
    return render_template('select_date.html', city=city, img_str=img_str)

@app.route('/plotting')
def plotting():
    city = request.args.get('city')
    return render_template('plotting.html', city=city)

@app.route('/prediction')
def prediction():
    city = request.args.get('city')
    return render_template('prediction.html', city=city)

manage_dataset()

if __name__ == '__main__':
    app.run(debug=True)