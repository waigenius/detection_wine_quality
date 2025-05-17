# Projet Python : Évaluation de la qualité du vin, sur une échelle de 0 à 10
Dans le cadre de notre formation au module Python, notre examen porte sur la création d'une application web (webapp) qui utilise un modèle de Machine Learning (ML), avec apprentissage supervisé, afin de répondre à une problématique donnée.

## L'objectif de la webapp
L'objectif de notre travail est de concevoir une webapp qui prédit un score de qualité pour un vin, à partir de ses paramètres physico-chimiques. Ce projet nous permet d'identifier les caractéristiques physico-chimiques qui influencent la qualité du vin.

Pour ce faire, nous avons utilisé un jeu de données (dataset) sur la qualité des vins en fonction de leurs caractéristiques physico-chimiques. Le dataset a été téléchargé depuis la plateforme Kaggle – Wine Quality Dataset. https://www.kaggle.com/datasets/yasserh/wine-quality-dataset/data

Il se compose de douze(12) colonnes, dont la variable cible *quality* et les  onze(11) caractéristiques physico-chimiques du vin qui sont:
- *fixed acidity*: acidité fixe
- *volatile acidity*: acidité volatile
- *citric acid*: acide citrique
- *residual sugar*: sucre résiduel
- *chlorides*: chlorures
- *free sulfur dioxide*: dioxyde de soufre libre
- *total sulfur dioxide*: dioxyde de soufre total
- *density*: densité
- *pH*
- *sulphates*: sulfates
- *alcohol*: alcool

Notre modèle apprend à reconnaître les caractéristiques importantes qui influencent le plus la qualité du vin, grâce à un apprentissage supervisé lors de la phase d'entraînement.

## Modèle de Machine Learning
Nous avons utilisé un modèle de régression supervisée, et plus précisément l'algorithme *RandomForestRegressor* de la librairie *Scikit-learn*. Le modèle entraîné est exporté au format model.joblib et est utilisé par la webapp.

Le taux d'erreur du modèle est de 46%. 

## Fonctionnement global de l'application
Pour lancer l'application, utilisez la commande suivante dans votre terminal :

``streamlit run app.py``

L'application offre une interface utilisateur avec des curseurs interactifs pour définir les paramètres du vin.
Pour chaque ensemble de paramètres, l'application fournit :

* Une estimation de la qualité du vin, via le modèle de ML.

* Un message personnalisé en fonction du score prédit.

* Une description automatique du vin, de type sommelier, basée sur le score.

* L'application inclut également des visualisations :

    - Un histogramme des scores de qualité du dataset, avec un repère visuel indiquant la prédiction pour les paramètres saisis par l'utilisateur.

    - Un graphique comparant le profil chimique du vin de l'utilisateur à la moyenne globale du dataset.

    - Une carte de chaleur des corrélations entre les caractéristiques physico-chimiques et la qualité du vin.

    - Une jauge dynamique représentant le score de qualité estimé.

En plus, nous avons mis un lien qui permet d'exporter les résultats de l'analyse au format CSV.

## Conclusion
En définitive, ce projet démontre comment un modèle de machine learning peut être intégré dans une application conviviale et pédagogique. L'application rend un modèle de ML plus accessible, interprétable et visuellement attrayant pour l'utilisateur final.