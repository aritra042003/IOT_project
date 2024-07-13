from flask import Flask, make_response, render_template, request, jsonify, session, redirect, url_for, flash, send_file
import sqlite3
import datetime
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import io

app = Flask(__name__)
app.config['DEBUG'] = True
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# Initialize variables to store temperature and humidity
latest_temperature = None
latest_humidity = None

# File to store collected sensor data
data_file = 'sensor_data.txt'

def write_data_to_file(data):
    with open(data_file, 'a') as file:
        file.write(f"Date: {data['date']}, Time: {data['time']}, Temperature: {data['temperature']}, Humidity: {data['humidity']}\n")

def create_table_if_not_exists():
    conn = sqlite3.connect('sensor_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sensor_data
                 (date TEXT, time TEXT, temperature REAL, humidity REAL)''')
    conn.commit()
    conn.close()

def create_rain_prediction_database():
    conn = sqlite3.connect('rain_prediction.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS rain_prediction
                 (date TEXT, time TEXT, temperature REAL, humidity REAL, prediction TEXT)''')
    conn.commit()
    conn.close()

def store_rain_prediction_to_new_db(data, prediction):
    conn = sqlite3.connect('rain_prediction.db')
    c = conn.cursor()
    c.execute("INSERT INTO rain_prediction (date, time, temperature, humidity, prediction) VALUES (?, ?, ?, ?, ?)",
              (data['date'], data['time'], data['temperature'], data['humidity'], prediction))
    conn.commit()
    conn.close()

def store_data_to_database(data):
    conn = sqlite3.connect('sensor_data.db')
    c = conn.cursor()
    c.execute("INSERT INTO sensor_data (date, time, temperature, humidity) VALUES (?, ?, ?, ?)",
              (data['date'], data['time'], data['temperature'], data['humidity']))
    conn.commit()
    conn.close()

def execute_query(query):
    conn = sqlite3.connect('sensor_data.db')
    c = conn.cursor()
    c.execute(query)
    result = c.fetchall()
    conn.close()
    return result

def create_user_table():
    conn = sqlite3.connect('user.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT, 
                 first_name TEXT, last_name TEXT, dob TEXT, country TEXT, phone TEXT)''')
    conn.commit()
    conn.close()

def register_user(username, password, first_name, last_name, dob, country, phone):
    conn = sqlite3.connect('user.db')
    c = conn.cursor()
    c.execute("INSERT INTO users (username, password, first_name, last_name, dob, country, phone) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (username, password, first_name, last_name, dob, country, phone))
    conn.commit()
    conn.close()

def authenticate_user(username, password):
    conn = sqlite3.connect('user.db')
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username=? AND password=?", (username, password))
    user = c.fetchone()
    conn.close()
    return user

@app.route('/', methods=['GET'])
def welcome():
    return render_template('welcome.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = authenticate_user(username, password)
        if user:
            session['user'] = username
            flash('Login successful', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'error')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        dob = request.form['dob']
        country = request.form['country']
        phone = request.form['phone']

        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('register'))

        conn = sqlite3.connect('user.db')
        c = conn.cursor()
        c.execute("SELECT id FROM users WHERE username=?", (username,))
        result = c.fetchone()
        conn.close()
        if result:
            flash('Username already exists', 'error')
            return redirect(url_for('register'))

        register_user(username, password, first_name, last_name, dob, country, phone)
        flash('Registration successful. You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/home')
def home():
    if 'user' in session:
        return render_template('home.html')
    else:
        flash('You need to login first', 'error')
        return redirect(url_for('login'))
@app.route('/plot/temp-humidity-over-time')
def plot_temp_humidity_over_time():
    conn = sqlite3.connect('sensor_data.db')
    df = pd.read_sql_query("SELECT date, temperature, humidity FROM sensor_data ORDER BY date", conn)
    conn.close()

    df['date'] = pd.to_datetime(df['date'])
    fig, ax = plt.subplots()
    ax.plot(df['date'], df['temperature'], label='Temperature', color='red')
    ax.plot(df['date'], df['humidity'], label='Humidity', color='blue')
    ax.set_xlabel('Date')
    ax.set_ylabel('Values')
    ax.set_title('Temperature and Humidity over Time')
    ax.legend()

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return make_response(output.getvalue(), 200, {'Content-Type': 'image/png'})
@app.route('/plot/temp-vs-humidity')
def plot_temp_vs_humidity():
    conn = sqlite3.connect('sensor_data.db')
    df = pd.read_sql_query("SELECT temperature, humidity FROM sensor_data", conn)
    conn.close()

    fig, ax = plt.subplots()
    ax.scatter(df['temperature'], df['humidity'], c='green')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Humidity')
    ax.set_title('Temperature vs Humidity')

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return make_response(output.getvalue(), 200, {'Content-Type': 'image/png'})
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/demo')
def demo():
    return render_template('demo.html')

@app.route('/sensor-data', methods=['POST'])
def receive_sensor_data():
    global latest_temperature, latest_humidity
    data = request.json
    latest_temperature = data.get('temperature')
    latest_humidity = data.get('humidity')

    now = datetime.datetime.now()
    data['date'] = now.strftime("%Y-%m-%d")
    data['time'] = now.strftime("%H:%M:%S")

    create_table_if_not_exists()
    create_rain_prediction_database()
    store_data_to_database(data)
    prediction = predict_rain(latest_temperature, latest_humidity)
    store_rain_prediction_to_new_db(data, prediction)
    write_data_to_file(data)

    return 'Data received successfully'

@app.route('/get-latest-sensor-data', methods=['GET'])
def get_latest_sensor_data():
    global latest_temperature, latest_humidity
    if latest_temperature is not None and latest_humidity is not None:
        return jsonify({'temperature': latest_temperature, 'humidity': latest_humidity})
    else:
        return 'No data available'

@app.route('/query', methods=['POST'])
def run_query():
    query = request.json['query']
    result = execute_query(query)
    return jsonify(result)

@app.route('/download-pdf', methods=['GET'])
def download_pdf():
    query = request.args.get('query')
    result = execute_query(query)
    if not result:
        return 'No data found'
    data = [list(row) for row in result]
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    table = Table(data)
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightyellow),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ])
    table.setStyle(style)
    col_widths = [max(len(str(row[i])) for row in data) * 12 for i in range(len(data[0]))]
    table._argW = col_widths
    doc.build([table])
    pdf_buffer.seek(0)
    return send_file(pdf_buffer, as_attachment=True, download_name='query_results.pdf', mimetype='application/pdf')

@app.route('/statistics', methods=['GET'])
def get_statistics():
    query = "SELECT AVG(temperature), AVG(humidity), MIN(temperature), MAX(temperature), MIN(humidity), MAX(humidity) FROM sensor_data"
    result = execute_query(query)
    avg_temperature, avg_humidity, min_temperature, max_temperature, min_humidity, max_humidity = result[0]
    return jsonify({
        'avg_temperature': avg_temperature,
        'avg_humidity': avg_humidity,
        'min_temperature': min_temperature,
        'max_temperature': max_temperature,
        'min_humidity': min_humidity,
        'max_humidity': max_humidity
    })

@app.route('/graph-data', methods=['GET'])
def get_graph_data():
    query = "SELECT date, temperature, humidity FROM sensor_data ORDER BY date"
    result = execute_query(query)
    labels = [row[0] for row in result]
    temperature_data = [row[1] for row in result]
    humidity_data = [row[2] for row in result]
    return jsonify({
        'labels': labels,
        'temperature_data': temperature_data,
        'humidity_data': humidity_data
    })

@app.route('/get-latest-rain-prediction', methods=['GET'])
def get_latest_rain_prediction():
    conn = sqlite3.connect('rain_prediction.db')
    c = conn.cursor()
    query = "SELECT * FROM rain_prediction ORDER BY date DESC, time DESC LIMIT 1"
    c.execute(query)
    result = c.fetchone()
    conn.close()
    if result:
        date, time, temperature, humidity, prediction = result
        return jsonify({
            'date': date,
            'time': time,
            'temperature': temperature,
            'humidity': humidity,
            'prediction': prediction
        })
    else:
        return jsonify({'error': 'No data available'})
    
@app.route('/profile')
def profile():
    if 'user' in session:
        return render_template('profile.html')
    else:
        flash('You need to login first', 'error')
        return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('You have been logged out', 'success')
    return redirect(url_for('welcome'))

def train_linear_regression_model():
    conn = sqlite3.connect('rain_prediction.db')
    df = pd.read_sql_query("SELECT temperature, humidity, prediction FROM rain_prediction", conn)
    conn.close()

    # Encoding prediction labels
    label_encoder = LabelEncoder()
    df['prediction'] = label_encoder.fit_transform(df['prediction'])

    X = df[['temperature', 'humidity']]
    y = df['prediction']

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predicting on the test set
    y_pred = model.predict(X_test)
    y_pred_class = np.round(y_pred).astype(int)

    # Calculating accuracy
    accuracy = accuracy_score(y_test, y_pred_class)
    return model, accuracy, label_encoder

@app.route('/train-model', methods=['GET'])
def train_model():
    model, accuracy, label_encoder = train_linear_regression_model()
    return jsonify({'accuracy': accuracy})

def predict_rain(temperature, humidity):
    model, accuracy, label_encoder = train_linear_regression_model()
    prediction = model.predict([[temperature, humidity]])
    prediction_class = np.round(prediction).astype(int)
    prediction_label = label_encoder.inverse_transform(prediction_class)[0]
    return prediction_label

if __name__ == '__main__':
    create_user_table()
    create_rain_prediction_database()
    app.run(host='0.0.0.0', port=5000)
