import pickle
from flask import Flask, request, jsonify
import numpy as np

# Load the models
with open('model1.bin', 'rb') as f:
    model = pickle.load(f)

with open('dv.bin', 'rb') as f:
    dv = pickle.load(f)

# Create a Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    client_data = request.json
    X_client = dv.transform([client_data])
    probability = model.predict_proba(X_client)[0][1]  # Probability of the positive class
    return jsonify({'probability': probability})

if __name__ == '__main__':

  app.run(debug=True, host='0.0.0.0', port=5000)  # Change to 0.0.0.0 to allow external access
