from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import os
import joblib

# Chemin vers les répertoires contenant les fichiers CSV pour chaque ligue
leagues_directories = {
    "bundesliga": "data/processed/bundesliga/",
    "laliga": "data/processed/laliga/",
    "ligue1": "data/processed/ligue1/",
    "premierleague": "data/processed/premierleague/",
    "seriea": "data/processed/seriea/"
}

# Boucle sur chaque ligue
for league, directory in leagues_directories.items():
    print(f"Entraînement du modèle pour la ligue {league}...")
    
    # Liste tous les fichiers CSV dans le répertoire de la ligue
    csv_files = [file for file in os.listdir(directory) if file.endswith(".csv")]
    
    # Boucle sur chaque fichier CSV dans la ligue
    for csv_file in csv_files:
        # Charger les données CSV
        df = pd.read_csv(os.path.join(directory, csv_file))
        
        # Afficher les premières lignes pour comprendre la structure des données
        print(f"Premières lignes du fichier {csv_file} :")
        print(df.head())
        
        # Sélectionner les fonctionnalités et la cible
        if league == "bundesliga":
            # La Bundesliga n'a pas de colonnes Points, Position, Club ou POS
            features = df.drop(columns=["MatchesM", "WonW", "DrawD", "LostL", "GoalsG", "+/-", "PointsP"])
            target = df["PointsP"]  # La cible est le nombre de points
        elif league == "laliga":
            # Colonnes dans le fichier laliga
            features = df.drop(columns=["#", "Équipe", "Points", "Pld", "GD", "W", "D", "L", "GF", "GA"])
            target = df["Points"]  # La cible est le nombre de points
        elif league == "ligue1":
            # Colonnes dans le fichier ligue1
            features = df.drop(columns=["Position", "Team", "Points", "Played", "Wins", "Draws", "Losses", "GF", "GA", "GD"])
            target = df["Points"]  # La cible est le nombre de points
        elif league == "premierleague":
            # Colonnes dans le fichier premierleague
            features = df.drop(columns=["Position", "Club", "Played", "Won", "Drawn", "Lost", "GF", "GA", "GD", "Points"])
            target = df["Points"]  # La cible est le nombre de points
        elif league == "seriea":
            # Colonnes dans le fichier seriea
            features = df.drop(columns=["POS", "Club", "PTS", "P", "W", "D", "L", "GF", "GA", "GD"])
            target = df["PTS"]  # La cible est le nombre de points
        else:
            raise ValueError(f"League {league} not supported")
        
        # Vérifier s'il y a des valeurs NaN avant de remplacer
        if features.isnull().values.any():
            features.fillna(0, inplace=True)

        
        # Division des données en ensembles d'entraînement et de test
        try:
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        except ValueError as e:
            print(f"Erreur lors de la division des données : {e}")
        
        # Vérifier les données avant de les passer au modèle
        print("Shapes of X_train and y_train:")
        print(X_train.shape, y_train.shape)
        
        # Création et entraînement du modèle
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Prédiction et évaluation du modèle
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f"Mean Squared Error for {csv_file}: {mse}")
        
        # Sauvegarde du modèle entraîné
        model_filename = f"{league}_{csv_file.split('.')[0]}_model.pkl"
        model_filepath = os.path.join("trained_models", model_filename)
        joblib.dump(model, model_filepath)
        
        print(f"Modèle pour {csv_file} sauvegardé avec succès.")
    
    print(f"Modèles pour la ligue {league} entraînés et sauvegardés avec succès.\n")
