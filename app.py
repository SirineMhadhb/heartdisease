import pandas as pd
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
# Importez votre modèle et les bibliothèques nécessaires
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Chargez votre modèle formé ici
model = LogisticRegression()  # Assurez-vous de charger votre modèle correctement

# Load your dataset
data = pd.read_csv('heart_disease_data.csv')


# Assuming 'target' is the column name containing the target variable
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Obtenez les données du formulaire
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        # Transform the input data similar to training data
        input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
        input_data = scaler.transform(input_data)  # Scale the input data

        # Make predictions
        prediction = model.predict(input_data)

        # Process the prediction result
        if prediction[0] == 0:
            result = 'Pas de maladie cardiaque'
        else:
            result = 'Maladie cardiaque détectée'

        return render_template('result.html', prediction=result)


if __name__ == '__main__':
    app.run(debug=True)

