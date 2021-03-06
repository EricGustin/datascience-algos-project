import os
from flask import Flask, jsonify, request
from mysklearn.myclassifiers import MyRandomForestClassifier
import joblib

# Working with the flask app and the API
app = Flask(__name__)



@app.route("/", methods=["GET"])
def index():
    return "<h1>Recidivism Predictor by Eric Gustin and Daniel Hoberman</h1>", 200

# This route is for the random song generating API
@app.route("/predictor", methods = ["GET"])
def predict():
    random_forest_classifier = joblib.load('trained_random_forest.pkl')
    prisoner_data = []
    temp_data = []

    for key, value in request.args.items():
        temp_data.append(value)
    prisoner_data.append(temp_data)
    print(prisoner_data)

    prediction = random_forest_classifier.predict(prisoner_data)
    if prediction is not None:
        return jsonify({"prediction": prediction}), 200
    else:
        return "Error making prediction", 400
    
  
# may have to adjust this depending on docker port
if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000.
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)
    app.run(debug=True)