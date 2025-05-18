# Projet Python : Évaluation de la qualité du vin, sur une échelle de 0 à 10

Dans le cadre de notre formation au module Python, notre examen porte sur la création d'une application web (webapp) à l'aide de la bibliothèque *Streamlit*, en utilisant un modèle d'apprentissage automatique  ou machine learning (ML) supervisé, afin de répondre à une problématique donnée.

## L'objectif de la webapp

Nous avons choisi d'étudier les caractéristiques physico-chimiques du vin afin d'évaluer sa qualité. Le vin représente une ressource significative pour l'économie des pays producteurs, non seulement en termes de revenus d'exportation, mais aussi en tant que vecteur de patrimoine culturel et de développement régional. L'étude de sa composition constitue donc une problématique majeure, qui peut aider les œnologues à tester diverses recettes ou compositions durant le processus de production.

Notre étude vise à identifier les constituants qui influencent la qualité du vin, afin de permettre aux producteurs d'optimiser leur production de vin.

Dans cette perspective, notre objectif est de concevoir une application web capable de prédire un score de qualité pour un vin, à partir de ses paramètres physico-chimiques.

Pour ce faire, nous avons utilisé un jeu de données (dataset) sur la qualité des vins en fonction de leurs caractéristiques physico-chimiques. Le dataset a été trouvé sur Kaggle via ce lien https://www.kaggle.com/discussions/getting-started/256014 , nous avons choisi le dataset sur le vin qui nous a redirigé sur l'addresse suivante https://archive.ics.uci.edu/dataset/186/wine+quality

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

Afin de traiter ce problème, nous avons utilisé la méthode de *Regression* pour prédire une valeur ordinale qui évaluera la qualité du vin. La variable cible est *quality*, de type entier inclu dans l'intervalle de 0 à 10.

Nous avons testé trois algorithmes de regressions à savoir *LinearRegressor*, *SVM* et *RandomForestRegressor* , présents dans la librairie *Scikit-learn*. L'algorithme qui a indiqué un meilleur score de performance parmis les trois est celui de *RandomForestRegressor* qui a été utilisé pour concevoir le modèle de ML. 

Le modèle entraîné est exporté au format model.joblib et est utilisé par la webapp.


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

Cette application peut servir dans le but d'améliorer et fabriquer de nouvelles saveurs de vin. Elle pourrait être utile à :
- des œnologues ou laboratoires pour évaluer rapidement un nouveau vin
- des étudiants en œnologie ou chimie alimentaire pour simuler des cas
- des producteurs ou caves pour comprendre l’impact des paramètres physico-chimiques sur la qualité perçue

Plus largement, elle démontre comment la data science peut aider à la prise de décision dans le domaine agroalimentaire, et assister par les modèles prédictifs accessibles à des non-spécialistes.

A l'issu de ce projet, nous réalisons comment un modèle de machine learning peut être intégré dans une application web pédagogique, conviviale, accessible, interprétable et visuellement attrayant pour l'utilisateur final.

