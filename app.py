import pandas as pd
from flask import Flask, render_template, request
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import joblib  # Importation de joblib pour enregistrer le modèle

app = Flask(__name__)

# Set the MLflow tracking URI (use the appropriate URL for your server)
mlflow.set_tracking_uri("http://localhost:5000")  # Replace with your MLflow tracking server URL

# Chargement du dataset
data = pd.read_csv('heart_disease_data.csv')

# Supposons que 'target' soit la colonne contenant la variable cible
X = data.drop('target', axis=1)
y = data['target']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mise à l'échelle des caractéristiques
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entraînement du modèle Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Exemple d'entrée (par exemple, une ligne de données d'entraînement)
example_input = X_train[0].reshape(1, -1)

# Enregistrement du modèle avec un exemple d'entrée et une signature
with mlflow.start_run() as run:
    mlflow.sklearn.log_model(model, "model", input_example=example_input)
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("accuracy", model.score(X_test, y_test))

    # Enregistrement du modèle dans un fichier .pkl pour DVC
    joblib.dump(model, 'model.pkl')  # Sauvegarde du modèle sous le nom 'model.pkl'
    print("Model saved as 'model.pkl'")

    # Accéder à l'ID de l'exécution en cours
    run_id = run.info.run_id  # Capture the run_id of the active run

# Imprimer l'ID de l'exécution pour vérification (optionnel)
print(f"Run ID: {run_id}")

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

        # Transformer les données d'entrée de manière similaire aux données d'entraînement
        input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
        input_data = scaler.transform(input_data)  # Mise à l'échelle des données d'entrée

        # Charger le modèle enregistré à partir du fichier .pkl
        model_loaded = joblib.load('model.pkl')  # Chargement du modèle depuis le fichier .pkl

        # Effectuer des prédictions
        prediction = model_loaded.predict(input_data)

        # Traiter le résultat de la prédiction
        if prediction[0] == 0:
            result = 'Pas de maladie cardiaque'
        else:
            result = 'Maladie cardiaque détectée'

        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
