# Utiliser une image Python officielle
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY . /app

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port pour l'API
EXPOSE 8000

# Démarrer le serveur FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
