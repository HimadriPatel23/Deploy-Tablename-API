import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify

# Load the Model
model = joblib.load('SVM_TrainedModel_07_03_Trainining.pkl')

# Create a Flask app
app = Flask(__name__)

# Create an API endpoint for predicting from excel inputs
@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    # Read the Excel file from the request
    file = request.files['file']
    data = pd.read_csv(file)

    from Testing.Test import test_model

    result = test_model(model,data,preprocess=True)

    # Save the predicted data to a new csv file
    #res.to_csv('predicted_data.csv', index=False)

    # Return the predicted data as a JSON object
    result_json = result.to_json(orient='records')
    return result_json

    # Return the prediction result as a JSON object
    #return jsonify({'predictions': list(y_pred)})

    #print(file)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)