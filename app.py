from flask import Flask, jsonify, render_template, request
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/select-city')
def select_city():
    return render_template('select_city.html')

@app.route('/city_list')
def city_list():
    df = pd.read_csv("./city_heat_index.csv")

    return jsonify(list(df["city"].unique()))

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