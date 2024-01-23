from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load the dataset
df = pd.read_csv("heart.csv")

# Split the dataset into features (X) and target variable (y)
X = df.drop('output', axis=1)
y = df.output

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html', X=X)

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form   
    features = [float(request.form.get(x)) for x in X.columns]
    input_data = [np.array(features)]

    # Make prediction using the pre-trained Linear Regression model
    linear_prediction = model.predict(input_data)

    # Initialize variables for logistic regression prediction and iteration
    logistic_prediction = None
    max_iterations = 100
    current_iteration = 0

    # Iterate until a valid prediction is found or max_iterations is reached
    while current_iteration < max_iterations:
        # Make prediction using the pre-trained Logistic Regression model
        logistic_prediction = lr.predict(input_data)

        # Check if the prediction is valid (you can define your own condition)
        if logistic_prediction[0] == 0 or logistic_prediction[0] == 1:
            break

        current_iteration += 1

    # Check if a valid prediction was obtained
    if logistic_prediction is not None:
        return render_template('index.html',
                               logistic_prediction=f'The Logistic Regression predicted output is: {logistic_prediction[0]}',
                               linear_prediction=f'The Linear Regression predicted output is: {linear_prediction[0]}', X=X)
    else:
        return render_template('index.html', error_message='Unable to obtain a valid logistic regression prediction', X=X)

if __name__ == '__main__':
    app.run(debug=True)
