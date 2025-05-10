# Script d'entraînement pour prédire la qualité d'un vin
# Modèle : RandomForestRegressor
# Données : dataset_wine.csv (avec une colonne 'quality' comme cible)

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Chargement des données
df = pd.read_csv("dataset_wine.csv", sep=",")

# Séparation des variables explicatives (X) et de la cible (y)
# On supprime également la colonne 'Id' qui n'a pas de valeur prédictive
X = df.drop(columns=["quality", "Id"])
y = df["quality"]

# Découpage des données en jeu d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialisation et entraînement du modèle
# Paramètres du modèle choisis pour un bon compromis performance/interprétabilité
model = RandomForestRegressor(
    n_estimators=200,   # nombre d'arbres
    max_depth=10,       # profondeur maximale des arbres
    random_state=42     # pour reproductibilité
)
model.fit(X_train, y_train)

# Évaluation du modèle sur le jeu de test
# Métriques utilisées : RMSE et R²
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("🔍 Évaluation du modèle :")
print(f" - RMSE : {mse**0.5:.2f}")
print(f" - R² : {r2:.2f}")
print(f" - Prédictions : min={y_pred.min():.2f}, max={y_pred.max():.2f}, écart-type={y_pred.std():.2f}")

# Étape 6 : Sauvegarde du modèle pour réutilisation dans Streamlit
joblib.dump(model, "model.joblib_test")
print("✅ Modèle sauvegardé dans 'model.joblib_test'")
