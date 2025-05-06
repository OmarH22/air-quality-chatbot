from flask import Flask, request, jsonify, send_file
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load saved model and encoders
rf_model = joblib.load('air_quality_rf_model.joblib')
encoders = joblib.load('air_quality_encoders.joblib')
le = encoders['Air_Quality']

# Optional: Load Decision Tree model (uncomment if using)
# dt_model = joblib.load('air_quality_dt_model.joblib')

@app.route('/')
def serve_index():
    return send_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Extract numerical inputs
        input_data = [
            data['temperature'],
            data['humidity'],
            data['pm25'],
            data['pm10'],
            data['no2'],
            data['so2'],
            data['co'],
            data['proximity'],
            data['population']
        ]
        # Predict using Random Forest
        prediction = rf_model.predict([input_data])[0]
        air_quality = le.inverse_transform([prediction])[0]
        return jsonify({'response': air_quality})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))  # Use Render's PORT or default to 5000
    app.run(host="0.0.0.0", port=port)  # Bind to 0.0.0.0 for Render