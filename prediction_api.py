import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from mysklearn.myclassifiers import MyRandomForestClassifier

# Working with the flask app and the API
app = Flask(__name__)
CORS(app)

interview_X_train = [
    ["Senior", "Java", "no", "no"],
    ["Senior", "Java", "no", "yes"],
    ["Mid", "Python", "no", "no"],
    ["Junior", "Python", "no", "no"],
    ["Junior", "R", "yes", "no"],
    ["Junior", "R", "yes", "yes"],
    ["Mid", "R", "yes", "yes"],
    ["Senior", "Python", "no", "no"],
    ["Senior", "R", "yes", "no"],
    ["Junior", "Python", "yes", "no"],
    ["Senior", "Python", "yes", "yes"],
    ["Mid", "Python", "no", "yes"],
    ["Mid", "Java", "yes", "no"],
    ["Junior", "Python", "no", "yes"],
]
interview_y_train = [
    "False",
    "False",
    "True",
    "True",
    "True",
    "False",
    "True",
    "False",
    "True",
    "True",
    "True",
    "True",
    "True",
    "False",
]

interview_X_test = [
    ["Junior", "Java", "yes", "no"],
    ["Junior", "Java", "yes", "yes"],
]

@app.route("/", methods=["GET"])
def index():
    return "<h1>Recidivism Predictor by Eric Gustin and Daniel Hoberman</h1>", 200

# This route is for the random song generating API
@app.route("/predictor", methods = ["GET"])
def predict():
    N = request.args.get("n", "")
    M = request.args.get("m", "")
    F = request.args.get("f", "")
    if(N == "" or M == "" or F == ""):
        N= 10
        M= 3
        F= 2
    random_forest_classifier = MyRandomForestClassifier(N, M, F)
    random_forest_classifier.fit(interview_X_train, interview_y_train)
    prediction = random_forest_classifier.predict(interview_X_test)
    if prediction is not None:
        return jsonify({"prediction": prediction.tolist()}), 200
    else:
        return "Error making prediction", 400
    
  
# may have to adjust this depending on docker port
if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
    app.run(debug=True)