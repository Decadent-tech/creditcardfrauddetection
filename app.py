#load random forest model to flask app

from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load Random Forest model
model = joblib.load('model/rf_fraud_model.pkl')  # Make sure the path is correct

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data['features']  # Expecting a list of 30 values

        # Reshape to 2D array for model input
        input_array = np.array(features).reshape(1, -1)

        # Predict
        prediction = model.predict(input_array)[0]
        probas = model.predict_proba(input_array)[0]

        return jsonify({
            'prediction': int(prediction),
            'probability_valid': round(probas[0], 4),
            'probability_fraud': round(probas[1], 4)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
