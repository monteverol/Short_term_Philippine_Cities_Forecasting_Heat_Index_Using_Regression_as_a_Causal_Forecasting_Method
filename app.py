from flask import Flask, jsonify, render_template, request
import io
import base64
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import urllib.parse
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import timedelta

matplotlib.use('agg')

app = Flask(__name__)

df = pd.read_csv("./city_heat_index.csv")

# REQUESTS
@app.route('/city_list')
def city_list():
    return jsonify(list(df["city"].unique()))

@app.route('/process_city', methods=['POST'])
def process_city():
    data = request.json
    city = data['city']

    city_data = df[df['city'] == city]
    features = ['Year', 'Month', 'Day']
    city_target = city_data['Average Heat Index ?C']

    city_data.loc[:, features] = city_data[features].ffill()

    X_train, X_test, y_train, y_test = train_test_split(city_data[features], city_target, test_size=0.1, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    y_train_scaled = y_train.copy()

    model = LinearRegression()
    model.fit(X_train_scaled, y_train_scaled)

    # Function to create and save the plot
    def create_plot():
        plt.scatter(X_train['Month'], y_train)
        plt.plot(X_train['Month'], model.predict(scaler.transform(X_train)), color='red')
        plt.title(f"Linear Regression for {city}")
        plt.xlabel('Month')
        plt.ylabel('Average Heat Index (°C)')
        
        # Save the plot as an image
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)

        # Convert the image to base64
        img_base64 = base64.b64encode(img.getvalue()).decode()
        img_str = f"data:image/png;base64,{img_base64}"
        return img_str

    img_str = create_plot()

    return jsonify({'message': 'City has been retrieved.', 'plot': img_str})

def clean_dataset():
    # Standardize date format
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    df.set_index('Date', inplace=True)
    df.drop(['Year', 'Month', 'Day'], axis=1, inplace=True)

    df['Month'] = df.index.month
    df['Day'] = df.index.day
    df['Year'] = df.index.year

    # Calculate the median for each month
    monthly_medians = df.replace(-999, pd.NA).groupby([df.index.year, df.index.month])['Average Heat Index ?C'].median()

    # Replace -999 with the monthly median
    for (year, month), median in monthly_medians.items():
        mask = (df.index.year == year) & (df.index.month == month)
        df.loc[mask, 'Average Heat Index ?C'] = df.loc[mask, 'Average Heat Index ?C'].replace(-999, median)

def predict(city):
    city_data = df[df['city'] == city]

    for i in range(1, 8):
        city_data = city_data.copy()
        city_data[f'lag_{i}'] = city_data['Average Heat Index ?C'].shift(i)

    # Split the dataset into features (X) and target variable (y)
    X = city_data.dropna()[['Month', 'Day', 'Year'] + [f'lag_{i}' for i in range(1, 8)]]
    y = city_data.dropna()['Average Heat Index ?C']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Train a model (you can use the same approach as in the process_city function)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    y_train_scaled = y_train.copy()

    model = LinearRegression()
    model.fit(X_train_scaled, y_train_scaled)

    # Use the trained model to predict the next 7 days
    current_date = X_test.index[-1]
    prediction_dates = [current_date + timedelta(days=i) for i in range(1, 8)]
    predictions = []

    for date in prediction_dates:
        # Create features for the prediction date
        prediction_features = [date.month, date.day, date.year]
        for i in range(1, 8):
            lagged_value = city_data.loc[date - timedelta(days=i), 'Average Heat Index ?C']
            prediction_features.append(lagged_value)

        # Scale the features
        prediction_features_scaled = scaler.transform([prediction_features])

        # Predict the Average Heat Index for the prediction date
        prediction = model.predict(prediction_features_scaled)
        predictions.append(prediction[0])

    # Return the predictions for the next 7 days
    return predictions

# FOR RENDER_TEMPLATE OF HTML
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/select-city')
def select_city():
    return render_template('select_city.html')

@app.route('/plotting')
def plotting():
    city = request.args.get('city')
    img_str = request.args.get('img_str')
    return render_template('plotting.html', city=city, img_str=img_str)

@app.route('/prediction')
def prediction():
    city = request.args.get('city')
    prediction_list = predict(city)
    return render_template('prediction.html', city=city, prediction_list=prediction_list)

clean_dataset()

# MAIN
if __name__ == '__main__':
    app.run(debug=True)