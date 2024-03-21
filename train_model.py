import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load the data
data = pd.read_csv('PremierLeague.csv')

# Preprocess the data
# Feature: FullTimeAwayTeamGoals
X = data['FullTimeAwayTeamGoals'].values.reshape(-1, 1)
y = data['FullTimeHomeTeamGoals'].values  # Target: FullTimeHomeTeamGoals

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, 'linear_regression_model.pkl')
