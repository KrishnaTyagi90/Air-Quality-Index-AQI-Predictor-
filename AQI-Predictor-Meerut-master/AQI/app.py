# Example using Flask for the backend

from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load('C:\\Users\\aayus\\AQI-Project-master\\AQI-Project-master\\random_forest_regression_model.pkl')

# Define a route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.get_json()

    # Perform any necessary preprocessing on the data

    # Make predictions using the loaded model
    prediction = model.predict(data['input'])

    # Return the prediction as JSON
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
