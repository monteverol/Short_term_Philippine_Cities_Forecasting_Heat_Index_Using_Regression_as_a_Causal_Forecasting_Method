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

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/select-city')
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

@app.route('/process_select_date', methods=['POST'])
def process_select_date():
    data = request.json
    date = data['date']
    
    print(date)
    return jsonify({ 'message': 'date retrieved', 'date': date })

@app.route('/select-date')
def select_date():
    return render_template('select_date.html')

@app.route('/plotting')
def plotting():
    city = request.args.get('city')
    return render_template('plotting.html', city=city)

@app.route('/prediction')
def prediction():
    city = request.args.get('city')
    return render_template('prediction.html', city=city)

if __name__ == '__main__':
    app.run(debug=True)