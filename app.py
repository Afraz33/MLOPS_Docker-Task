from flask import Flask, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('linear_regression_model.pkl')


@app.route('/predict', methods=['GET'])
def predict():
    # Dummy data for FullTimeAwayTeamGoals
    full_time_away_goals = 1

    # Make prediction
    prediction = model.predict([[full_time_away_goals]])

    return jsonify({'predicted_home_goals': prediction[0]})


if __name__ == '__main__':
    app.run(debug=True)
