import pickle
import logging
from waitress import serve
from flask import Flask
from flask import request
from flask import jsonify
logging.basicConfig(level=logging.INFO)

output_file = 'model_.bin'

# Load the model

with open(output_file, 'rb') as f_in: # replace w with r for reading
    dv, model = pickle.load(f_in)

app = Flask('depression_check')

@app.route ('/predict', methods=['POST'])
def predict():
    student = request.get_json()
    X_student = dv.transform(student)
    y_pred = model.predict_proba(X_student)[0, 1]
    depression = (y_pred >= 0.5)

    result = {
        'depression_probability': float(y_pred),
        'is_depressed': bool(depression),
    }
    return jsonify(result)

if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=9696)