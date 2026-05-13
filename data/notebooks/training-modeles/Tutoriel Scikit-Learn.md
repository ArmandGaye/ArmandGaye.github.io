# Python Machine Learning : Tutoriel Scikit-Learn

Ce tutoriel couvre les bases de l'apprentissage automatique en utilisant la bibliothèque Scikit-Learn en Python. Nous allons explorer les étapes clés du processus d'apprentissage automatique, y compris : - la préparation des données, 
- la sélection du modèle, 
- l'entraînement, et 
- l'évaluation.


## Préparation des données
La première étape de tout projet d'apprentissage automatique est la préparation des données. Cela inclut le nettoyage des données, la gestion des valeurs manquantes, et la transformation des données en un format approprié pour les modèles d'apprentissage automatique.
```python
import pandas as pd 
from sklearn.model_selection import train_test_split
# Charger les données
data = pd.read_csv('data.csv') 
# Nettoyer les données (par exemple, gérer les valeurs manquantes)
data.fillna(data.mean(), inplace=True)
# Séparer les caractéristiques (X) et la cible (y)
X = data.drop('target', axis=1)
y = data['target']
# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Sélection du modèle
Après avoir préparé les données, la prochaine étape est de sélectionner un modèle d'apprentissage automatique approprié. Scikit-Learn offre une variété de modèles pour les tâches de classification, de régression, et de clustering.
```python
from sklearn.ensemble import RandomForestClassifier
# Sélectionner un modèle (par exemple, Random Forest pour la classification)
model = RandomForestClassifier(n_estimators=100, random_state=42)
```

## Entraînement du modèle
Une fois que nous avons sélectionné notre modèle, nous pouvons l'entraîner sur les données d'entraînement.
```python
# Entraîner le modèle
model.fit(X_train, y_train)
```

## Évaluation du modèle
Après l'entraînement, il est crucial d'évaluer les performances du modèle sur les données de test pour comprendre sa capacité à généraliser à de nouvelles données.
```python
from sklearn.metrics import accuracy_score, classification_report
# Faire des prédictions sur les données de test
y_pred = model.predict(X_test)
# Évaluer les performances du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision : {accuracy}")
print("Rapport de classification :")
print(classification_report(y_test, y_pred))
```

Ce tutoriel fournit une introduction de base à l'utilisation de Scikit-Learn pour l'apprentissage automatique en Python. En suivant ces étapes, vous pouvez construire et évaluer des modèles d'apprentissage automatique pour diverses tâches.
