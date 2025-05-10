import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 🔹 Charger les données
df = pd.read_csv("dataset_wine.csv", sep=",")

# 🔹 Séparer les features (X) et la cible (y)
X = df.drop(columns=["quality", "Id"])
y = df["quality"]

# 🔹 Diviser en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔹 Créer et entraîner le modèle
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# 🔹 Évaluer sur le test set
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("🔍 Évaluation du modèle :")
print(f" - RMSE : {mse**0.5:.2f}")
print(f" - R² : {r2:.2f}")
print(f" - Prédictions : min={y_pred.min():.2f}, max={y_pred.max():.2f}, écart-type={y_pred.std():.2f}")

# 🔹 Sauvegarder le modèle
joblib.dump(model, "model.joblib_test")
print("✅ Modèle sauvegardé dans 'model.joblib_test'")
