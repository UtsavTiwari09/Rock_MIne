import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, render_template, request

# Load the dataset
sonar_data = pd.read_csv('path_to_your_copy.csv', header=None)

# Preprocess the data and train the model
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60].map({'R': 0, 'M': 1})

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train RandomForest model
best_model = RandomForestClassifier(random_state=42, n_estimators=100)
best_model.fit(X_scaled, Y)

# Save the trained model and scaler to disk
with open('model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from user form
        input_data = request.form['input_data']
        input_data = list(map(float, input_data.split(',')))

        if len(input_data) != 60:
            return "Error: Please enter exactly 60 features."

        # Load model and scaler from disk
        with open('model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)

        # Process input data and make prediction
        input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
        input_data_scaled = scaler.transform(input_data_as_numpy_array)
        prediction = model.predict(input_data_scaled)
        prediction_label = 'Rock' if prediction[0] == 0 else 'Mine'

        return f"The object is classified as: {prediction_label}"

    except ValueError:
        return "Invalid input. Please enter numeric values separated by commas."

if __name__ == '__main__':
    app.run(debug=True)
