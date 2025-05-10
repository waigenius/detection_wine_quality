# WebApp Streamlit pour prédire la qualité d'un vin à partir de ses
# caractéristiques chimiques, à l'aide d'un modèle RandomForest entraîné.

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Chargement du modèle et du dataset
@st.cache_resource
def load_model():
    return joblib.load("model.joblib_test")

@st.cache_data
def load_dataset():
    return pd.read_csv("dataset_wine.csv", sep=",")  # ou winequality-red.csv

model = load_model()
df = load_dataset()
features = [col for col in df.columns if col not in ["quality", "Id"]]

# Configuration de l'interface
st.title("🍷 Prédisez la qualité de votre vin")
st.markdown("Ajustez les caractéristiques chimiques à gauche et visualisez le résultat à droite.")

# Fonction de description sommelier
def sommelier_note(alcohol, acidity, sugar):
    if alcohol > 13 and sugar < 3:
        return "Ce vin serait sec, corsé et riche en alcool – typé Bordeaux."
    elif sugar > 6:
        return "Vin doux et fruité – probablement un moelleux ou liquoreux."
    elif acidity > 0.7:
        return "Vin vif et nerveux – parfait pour les fruits de mer."
    else:
        return "Vin équilibré, agréable au palais, sans excès marqué."

# --- Mise en page en deux colonnes
col_inputs, col_output = st.columns([2, 3])

with col_inputs:
    st.subheader("🔬 Caractéristiques du vin")
    col1, col2 = st.columns(2)
    
    # Sliders pour chaque variable d'entrée (groupés en 2 colonnes)
    with col1:
        fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 7.0, 0.1)
        citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.3, 0.01)
        residual_sugar = st.slider("Residual Sugar", 0.5, 16.0, 2.0, 0.1)
        free_sulfur_dioxide = st.slider("Free SO₂", 1, 72, 15, 1)
        pH = st.slider("pH", 2.8, 4.2, 3.2, 0.01)
        alcohol = st.slider("Alcohol (%)", 8.0, 15.0, 10.0, 0.1)
    with col2:
        volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.5, 0.5, 0.01)
        chlorides = st.slider("Chlorides", 0.01, 0.2, 0.05, 0.005)
        total_sulfur_dioxide = st.slider("Total SO₂", 6, 289, 46, 1)
        sulphates = st.slider("Sulphates", 0.2, 2.0, 0.5, 0.01)
        density = st.slider("Density", 0.9900, 1.0050, 0.9960, 0.0001)

# Construction de la ligne d'entrée utilisateur sous forme de DataFrame
user_input = pd.DataFrame([{
    'fixed acidity': fixed_acidity,
    'volatile acidity': volatile_acidity,
    'citric acid': citric_acid,
    'residual sugar': residual_sugar,
    'chlorides': chlorides,
    'free sulfur dioxide': free_sulfur_dioxide,
    'total sulfur dioxide': total_sulfur_dioxide,
    'density': density,
    'pH': pH,
    'sulphates': sulphates,
    'alcohol': alcohol
}])

# PRÉDICTION ET AFFICHAGE DES RÉSULTATS
with col_output:
    st.subheader("📈 Résultats")
    if st.button("🍷 Prédire la qualité du vin"):
        prediction = model.predict(user_input)[0]
        prediction = round(prediction, 2)
        st.metric(label="Qualité estimée", value=f"{prediction} / 10")

        # Message adapté selon la note prédite
        if prediction >= 7:
            st.success("Excellent vin 🍾 – digne d'un sommelier !")
        elif prediction >= 6:
            st.info("Bon vin 👍 – agréable au palais.")
        elif prediction >= 5:
            st.warning("Vin correct 😐 – mais peut mieux faire.")
        else:
            st.error("Vin de qualité médiocre 😬 – à revoir.")

        # Affichage de la description "sommelier"
        st.markdown(f"📜 **Note de dégustation** : *{sommelier_note(alcohol, fixed_acidity, residual_sugar)}*")

        # VISUALISATIONS EN TABS
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Histogramme", "📉 Comparatif", "🧭 Corrélations", "🎯 Jauge"])

        # Histogramme de la qualité des vins
        with tab1:
            fig = px.histogram(df, x="quality", nbins=10, title="Répartition des scores qualité")
            fig.add_vline(x=prediction, line_color="red", line_dash="dash")
            st.plotly_chart(fig, use_container_width=True)

        # Comparaison de profil utilisateur vs moyenne dataset
        with tab2:
            mean_values = df[features].mean()
            user_vs_mean = pd.DataFrame({
                "Caractéristique": features,
                "Votre vin": user_input.iloc[0].values,
                "Moyenne": mean_values.values
            })
            df_melted = user_vs_mean.melt(id_vars="Caractéristique", var_name="Type", value_name="Valeur")
            fig = px.bar(df_melted, x="Caractéristique", y="Valeur", color="Type", barmode="group", title="Comparaison avec la moyenne des vins")
            st.plotly_chart(fig, use_container_width=True)

        # Corrélation entre qualité et variables
        with tab3:
            fig, ax = plt.subplots(figsize=(10, 6))
            corr = df.corr()
            sns.heatmap(corr[['quality']].sort_values(by='quality', ascending=False), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        # Jauge de qualité prédite
        with tab4:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction,
                title={'text': "Qualité estimée"},
                gauge={
                    'axis': {'range': [0, 10]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 5], 'color': "lightgray"},
                        {'range': [5, 7], 'color': "gold"},
                        {'range': [7, 10], 'color': "lightgreen"}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

        # Export CSV avec prédiction
        csv = user_input.copy()
        csv["predicted_quality"] = prediction
        st.download_button(
            label="📥 Télécharger les données (CSV)",
            data=csv.to_csv(index=False).encode('utf-8'),
            file_name='mon_vin_personnalise.csv',
            mime='text/csv',
        )
