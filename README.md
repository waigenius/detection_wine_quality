# Evaluation de la qualité du vin, sur une echelle de 0 à 10

Lancer l'app avec cette ligne dans le terminal : streamlit run app.py

Dans le cadre de notre apprentissage dans le module Python, notre evaluation porte sur la création d'une webapp en utilisant un modèle de ML, avec l'apprentissage supervisée afin de répondre à une problématique donnée.


## L'objectif de la webapp
L'objectif de notre travail est de concevoir une webapp  qui va prédire un score sur la qualité du vin selon les paramètres physico-chimiques fournies. A l'issu de notre nous permettre de connaitre les caractéristiques qui influencent sur la qualité du vin.


## A propos du Dataset
Le dataset utilisé a été téléchargé sur la plateforme Kaggle – Wine Quality Dataset. Il se compose de 13 colonnes dont 11 sont les caractéristiques et les deux autres sont le score associé du vin et Id.


## Modele de Machine Learning
Il s'agit d'une modèle de regression supervisée. Algorithme RandomForestRegressor de Scikit-learn. Modèle exporté sous model.joblib et utilisé par la WebApp.

## Le fonctionnement global de l'application
Entrées utilisateur grâce aux sliders intéractifs pour définir les paramètres du vin. Pour la prédisction, nous avons une éstimation de la qualité via le modèle ML, message personnalisé en fonction du score et description sommelier automatique. Pour la partie visualisation, nous avons un histogramme de scores du dataset avec repère utilisateur, comparaison du profil chimique utilisateurs versus moyenne globale, carte de chaleur des corrélations avec la qualité et Jauge dynamique du score estimé. Une fonction d'export en format CSV en bonus.

## Conclusion
Ce projet montre comment un modèle de machine learning peut être intégré dans une application conviviale et pédagogique.
Il illustre l'intérêt de l'interactivité pour rendre un modèle plus accessible, interprétable et visuellement engageant pour l'utilisateur final.