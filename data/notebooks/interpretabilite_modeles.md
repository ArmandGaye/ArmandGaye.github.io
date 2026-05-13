# 🔍 Interprétabilité des Modèles de Machine Learning
## Du "ça prédit bien" au "je comprends pourquoi"

> **Public cible** : Data scientists ayant déjà entraîné des modèles et souhaitant aller au-delà des métriques de performance.  
> **Dataset utilisé** : Adult Income (UCI) — accessible via `sklearn`, prédit si le revenu annuel dépasse 50 000 $  
> **Outils** : `scikit-learn`, `shap`, `pandas`, `matplotlib`, `seaborn`

---

## Table des matières

1. [Pourquoi l'interprétabilité est un enjeu business ?](#1-pourquoi)
2. [Setup & chargement du dataset](#2-setup)
3. [Construction du modèle de base](#3-modele)
4. [Feature Importance — la version naïve et ses pièges](#4-feature-importance)
5. [SHAP Values — expliquer chaque prédiction](#5-shap)
6. [Analyse des cas où le modèle se trompe](#6-erreurs)
7. [Limites du modèle — ce qu'aucun dashboard ne montre](#7-limites)
8. [Checklist interprétabilité pour un contexte business](#8-checklist)

---

## 1. Pourquoi l'interprétabilité est un enjeu business ? {#1-pourquoi}

### Le piège du "95% d'accuracy"

Imaginez ce scénario : vous livrez un modèle de scoring crédit avec 95% d'accuracy. Le métier valide. Six mois plus tard, un client attaque l'entreprise en justice parce qu'il s'est vu refuser un prêt sans explication.

> **Problème** : vous ne savez pas pourquoi votre modèle a dit "non".

Ce n'est pas hypothétique. En Europe, le **RGPD (Article 22)** impose un droit à l'explication pour toute décision automatisée impactant une personne. Aux États-Unis, l'**Equal Credit Opportunity Act** interdit les discriminations dans l'octroi de crédit.

### Les vraies questions que pose le métier

| Ce que demande le data scientist | Ce que veut vraiment le métier |
|----------------------------------|-------------------------------|
| "Mon AUC est de 0.91" | "Quels clients sont à risque et pourquoi ?" |
| "Le modèle est bon sur le test set" | "Est-ce qu'il sera encore bon dans 6 mois ?" |
| "J'ai utilisé 47 features" | "Quelles variables ont un impact métier réel ?" |
| "L'accuracy est de 93%" | "Dans quels cas il se trompe et est-ce grave ?" |

### Trois niveaux d'interprétabilité

```
Niveau 1 — GLOBALE  : Comprendre le comportement général du modèle
                      → "En moyenne, quelle variable pèse le plus ?"

Niveau 2 — LOCALE   : Expliquer une prédiction individuelle
                      → "Pourquoi ce client spécifique est classé à risque ?"

Niveau 3 — CONTRASTIVE : Comprendre ce qui aurait changé la prédiction
                          → "Que faudrait-il changer pour obtenir un prêt ?"
```

---

## 2. Setup & chargement du dataset {#2-setup}

### Installation des dépendances

```bash
pip install shap scikit-learn pandas numpy matplotlib seaborn
```

### Chargement du dataset Adult Income

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import shap
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DU DATASET
# Source : UCI Machine Learning Repository
# 48 842 individus | 14 features | Cible : revenu > 50K$ (binaire)
# ─────────────────────────────────────────────────────────────────────────────

data = fetch_openml(name='adult', version=2, as_frame=True)
df = data.frame.copy()

print(f"Shape : {df.shape}")
print(f"\nDistribution de la cible :")
print(df['class'].value_counts(normalize=True).round(3))
```

**Output attendu :**
```
Shape : (48842, 15)

Distribution de la cible :
<=50K    0.761
>50K     0.239
```

> 📌 **Note pédagogique** : Le dataset est **déséquilibré** (76% vs 24%). Si votre modèle prédit toujours "<=50K", il obtient 76% d'accuracy. C'est pour ça que l'accuracy seule ne suffit pas.

### Préparation des données

```python
# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

# Séparer features et cible
X = df.drop(columns=['class'])
y = (df['class'] == '>50K').astype(int)

# Identifier les types de colonnes
categorical_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
numerical_cols   = X.select_dtypes(include=['number']).columns.tolist()

print(f"Features numériques  ({len(numerical_cols)}) : {numerical_cols}")
print(f"Features catégorielles ({len(categorical_cols)}) : {categorical_cols}")

# Encodage simple pour RandomForest
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

# Imputation + encodage
preprocessor = ColumnTransformer(transformers=[
    ('num', SimpleImputer(strategy='median'), numerical_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ]), categorical_cols)
])

# Noms des colonnes après transformation (utile pour les graphes)
feature_names = numerical_cols + categorical_cols

# Split train/test (stratifié pour respecter la distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Transformation
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)

# Conversion en DataFrame pour garder les noms de colonnes
X_train_df = pd.DataFrame(X_train_proc, columns=feature_names)
X_test_df  = pd.DataFrame(X_test_proc,  columns=feature_names)

print(f"\nTrain size : {X_train_df.shape[0]:,} | Test size : {X_test_df.shape[0]:,}")
```

---

## 3. Construction du modèle de base {#3-modele}

```python
# ─────────────────────────────────────────────────────────────────────────────
# ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────────────────────────

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=20,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_df, y_train)

# ─────────────────────────────────────────────────────────────────────────────
# ÉVALUATION — Au-delà de l'accuracy
# ─────────────────────────────────────────────────────────────────────────────

y_pred      = model.predict(X_test_df)
y_pred_prob = model.predict_proba(X_test_df)[:, 1]

print("=" * 60)
print("RAPPORT DE CLASSIFICATION")
print("=" * 60)
print(classification_report(y_test, y_pred,
      target_names=['<=50K', '>50K']))

print(f"AUC-ROC : {roc_auc_score(y_test, y_pred_prob):.4f}")
```

**Output attendu :**
```
============================================================
RAPPORT DE CLASSIFICATION
============================================================
              precision    recall  f1-score   support

       <=50K       0.90      0.94      0.92      7408
        >50K       0.77      0.65      0.71      2321

    accuracy                           0.88      9729
   macro avg       0.84      0.80      0.81      9729
weighted avg       0.87      0.88      0.87      9729

AUC-ROC : 0.9241
```

> ⚠️ **Observation critique** : Le recall sur ">50K" est de 65%. Le modèle **rate 35% des personnes qui gagnent plus de 50K$**. Est-ce acceptable dans votre contexte métier ? Si ce modèle sert à cibler une campagne marketing premium, vous perdez 35% de votre audience cible. La métrique seule ne vous dit pas ça.

---

## 4. Feature Importance — la version naïve et ses pièges {#4-feature-importance}

### 4.1 L'importance par défaut de scikit-learn (MDI)

```python
# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE NAÏVE (Mean Decrease in Impurity)
# ─────────────────────────────────────────────────────────────────────────────

fi_df = pd.DataFrame({
    'feature':    feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=fi_df, x='importance', y='feature', palette='viridis')
plt.title('Feature Importance (MDI — Random Forest)', fontsize=14, fontweight='bold')
plt.xlabel('Importance moyenne')
plt.tight_layout()
plt.savefig('feature_importance_mdi.png', dpi=150)
plt.show()

print(fi_df.to_string(index=False))
```

**Interprétation commentée :**
```
       feature  importance
 capital.gain    0.2341   ← Très forte : les gains en capital distinguent bien
       fnlwgt    0.1187   ← ⚠️ ATTENTION : variable de pondération census, 
                              pas une vraie feature prédictive !
  education.num  0.1053   ← Niveau d'éducation en années
           age    0.0987   ← L'âge joue un rôle important
  hours.per.week 0.0876
  capital.loss   0.0654
  ...
```

> 🚨 **Piège #1 — Le biais des variables continues** : Les variables avec beaucoup de valeurs uniques (`fnlwgt`, `age`, `capital.gain`) gonflent artificiellement leur MDI importance dans les arbres. Ce n'est pas de la "vraie" importance — c'est un artefact mathématique.

> 🚨 **Piège #2 — Les variables corrélées** : Si `education` et `education.num` sont corrélées (elles le sont !), leur importance se partage. Individuellement elles semblent moins importantes qu'elles ne le sont réellement.

### 4.2 Permutation Importance — plus honnête

```python
from sklearn.inspection import permutation_importance

# ─────────────────────────────────────────────────────────────────────────────
# PERMUTATION IMPORTANCE
# Principe : on mélange aléatoirement une colonne et on mesure la dégradation
#            de la performance. Si mélanger une colonne ne change rien → 
#            elle n'est pas importante.
# ─────────────────────────────────────────────────────────────────────────────

perm_imp = permutation_importance(
    model, X_test_df, y_test,
    n_repeats=10,        # 10 permutations pour avoir une variance
    random_state=42,
    scoring='roc_auc',
    n_jobs=-1
)

perm_df = pd.DataFrame({
    'feature': feature_names,
    'importance_mean': perm_imp.importances_mean,
    'importance_std':  perm_imp.importances_std
}).sort_values('importance_mean', ascending=False)

# Visualisation avec barres d'erreur (la variance compte !)
plt.figure(figsize=(10, 6))
plt.barh(perm_df['feature'], perm_df['importance_mean'],
         xerr=perm_df['importance_std'],
         color='steelblue', alpha=0.8, ecolor='gray', capsize=4)
plt.xlabel('Diminution AUC-ROC (moyenne ± std)')
plt.title('Permutation Importance sur le test set', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('permutation_importance.png', dpi=150)
plt.show()
```

> 📌 **Ce que ça change** : `fnlwgt` apparaît beaucoup moins importante en permutation importance. La MDI nous mentait. La variable `capital.gain` reste forte dans les deux cas — là on peut avoir confiance.

**Règle d'or** : utilisez TOUJOURS la permutation importance sur le **test set** pour valider ce que vous dit la MDI.

---

## 5. SHAP Values — expliquer chaque prédiction {#5-shap}

SHAP (SHapley Additive exPlanations) est basé sur la théorie des jeux coopératifs. L'idée : **chaque feature "contribue" à déplacer la prédiction depuis la valeur moyenne** (baseline) vers la prédiction finale.

```
Prédiction finale = valeur_baseline + contribution(age) + contribution(education) + ...
```

### 5.1 Calcul des SHAP values

```python
# ─────────────────────────────────────────────────────────────────────────────
# CALCUL DES SHAP VALUES
# TreeExplainer est optimisé pour les modèles à base d'arbres (RF, XGBoost...)
# ─────────────────────────────────────────────────────────────────────────────

explainer  = shap.TreeExplainer(model)

# Sur un sous-ensemble pour la rapidité (1000 exemples suffisent pour les viz globales)
sample_idx = np.random.choice(len(X_test_df), size=1000, replace=False)
X_sample   = X_test_df.iloc[sample_idx]

shap_values = explainer.shap_values(X_sample)

# shap_values est une liste de 2 éléments (classe 0 et classe 1)
# On prend la classe 1 (>50K)
shap_vals_class1 = shap_values[1]

print(f"Shape des SHAP values : {shap_vals_class1.shape}")
# → (1000, 14) — une valeur SHAP par individu par feature
```

### 5.2 Graphe de synthèse global (Beeswarm plot)

```python
# ─────────────────────────────────────────────────────────────────────────────
# BEESWARM PLOT — Le graphe SHAP le plus informatif
# Chaque point = un individu
# Position horizontale = contribution SHAP (positive → vers ">50K")
# Couleur = valeur de la feature (rouge = haute, bleu = basse)
# ─────────────────────────────────────────────────────────────────────────────

plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_vals_class1,
    X_sample,
    feature_names=feature_names,
    plot_type='dot',
    show=False
)
plt.title('SHAP Summary Plot — Impact sur la prédiction ">50K"', fontsize=13)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Comment lire ce graphe — guide ligne par ligne :**

```
capital.gain  ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░
  → Points rouges à droite : une valeur élevée de capital.gain
    pousse FORTEMENT vers ">50K"
  → Points bleus regroupés près de 0 : la plupart des gens ont
    capital.gain = 0, donc ça ne change rien pour eux.

age           ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  → Gradient rouge→bleu de droite→gauche : être âgé (+rouge)
    augmente la probabilité ">50K", être jeune la diminue.
  → C'est une RELATION MONOTONE lisible directement.

education.num ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  → Même logique : plus d'années d'études = plus de chances.

fnlwgt        ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  → Distribution étalée sans tendance claire rouge/bleu.
  → ⚠️ Cela confirme notre suspicion : fnlwgt n'a pas de
    signal directionnel = c'était du bruit dans la MDI.

sex           ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  → Deux groupes distincts : ça suggère un encodage 0/1
  → À analyser attentivement pour les biais !
```

### 5.3 SHAP Dependence Plot — relation entre feature et prédiction

```python
# ─────────────────────────────────────────────────────────────────────────────
# DEPENDENCE PLOT
# Montre comment la valeur d'une feature affecte sa contribution SHAP
# La couleur représente l'interaction avec une autre feature (choisie auto)
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Feature 1 : age
plt.sca(axes[0])
shap.dependence_plot(
    'age', shap_vals_class1, X_sample,
    feature_names=feature_names,
    ax=axes[0], show=False
)
axes[0].set_title('SHAP : impact de l\'âge', fontweight='bold')

# Feature 2 : education.num
plt.sca(axes[1])
shap.dependence_plot(
    'education.num', shap_vals_class1, X_sample,
    feature_names=feature_names,
    ax=axes[1], show=False
)
axes[1].set_title('SHAP : impact de l\'éducation', fontweight='bold')

plt.tight_layout()
plt.savefig('shap_dependence.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Interprétation du Dependence Plot pour l'âge :**
```
- Entre 20 et 30 ans   : contribution négative (SHAP < 0)
  → "Être jeune" diminue la probabilité d'avoir >50K

- Entre 30 et 50 ans   : contribution croissante et positive
  → La courbe de salaire classique

- Après 60 ans         : légère décroissance
  → Peut-être due aux retraités dans le dataset

- La couleur (souvent education.num) montre les interactions :
  un salarié de 45 ans avec peu d'éducation (bleu) reste moins
  bien classé qu'un cadre de 45 ans avec beaucoup d'éducation (rouge)
```

### 5.4 SHAP local — expliquer une prédiction individuelle

```python
# ─────────────────────────────────────────────────────────────────────────────
# WATERFALL PLOT — "Pourquoi CE client a-t-il été classé ainsi ?"
# C'est la réponse à la question business individuelle
# ─────────────────────────────────────────────────────────────────────────────

# Sélection d'un exemple "positif" (prédit >50K avec haute confiance)
high_income_idx = np.where(
    (model.predict_proba(X_test_df)[:, 1] > 0.85) & (y_test.values == 1)
)[0][0]

# Profil de cet individu
print("PROFIL DE L'INDIVIDU :")
print(X_test_df.iloc[high_income_idx].to_frame().T.to_string())
print(f"\nProbabilité prédite >50K : {model.predict_proba(X_test_df)[high_income_idx, 1]:.2%}")
print(f"Vraie étiquette          : {'>50K' if y_test.values[high_income_idx] == 1 else '<=50K'}")
```

**Output type :**
```
PROFIL DE L'INDIVIDU :
   age  fnlwgt  education.num  capital.gain  hours.per.week  sex
    52  287927             16         14084              50    1

Probabilité prédite >50K : 96.3%
Vraie étiquette          : >50K
```

```python
# Explanation individuelle
shap_single = explainer.shap_values(X_test_df.iloc[[high_income_idx]])[1][0]

# Waterfall plot (SHAP moderne)
explanation = shap.Explanation(
    values      = shap_single,
    base_values = explainer.expected_value[1],
    data        = X_test_df.iloc[high_income_idx].values,
    feature_names = feature_names
)

plt.figure(figsize=(10, 5))
shap.waterfall_plot(explanation, show=False)
plt.title(f'Explication individuelle — P(>50K) = 96.3%', fontweight='bold')
plt.tight_layout()
plt.savefig('shap_waterfall.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Comment lire le Waterfall Plot :**
```
E[f(x)] = 0.245   ← Point de départ : la probabilité moyenne dans le dataset

+ capital.gain contribue +0.38  ← Ce gain de 14 084$ est le facteur #1
+ education.num contribue +0.15 ← 16 ans d'études (master/doctorat)
+ age contribue +0.08           ← 52 ans, expérience établie
+ hours.per.week contribue +0.05 ← 50h/semaine, investissement élevé
- fnlwgt contribue -0.02        ← Légère contribution négative (bruit)
────────────────────────────────
f(x) = 0.963                   ← Probabilité finale prédite
```

> 💡 **Valeur business** : Vous pouvez maintenant répondre à "pourquoi ce client a été classé premium" avec des éléments factuels, auditables, et compréhensibles par un non-technicien.

### 5.5 Force Plot — visualisation alternative

```python
# ─────────────────────────────────────────────────────────────────────────────
# FORCE PLOT — version compacte, idéale pour les dashboards
# ─────────────────────────────────────────────────────────────────────────────

shap.initjs()  # Nécessaire pour le rendu HTML interactif

force_plot = shap.force_plot(
    base_value    = explainer.expected_value[1],
    shap_values   = shap_single,
    features      = X_test_df.iloc[high_income_idx],
    feature_names = feature_names
)
shap.save_html('force_plot.html', force_plot)
print("Force plot sauvegardé dans force_plot.html")
```

---

## 6. Analyse des cas où le modèle se trompe {#6-erreurs}

C'est l'étape la plus négligée et pourtant la plus précieuse. Les erreurs du modèle sont une **mine d'or d'informations**.

### 6.1 Construction du dataframe d'erreurs

```python
# ─────────────────────────────────────────────────────────────────────────────
# DATAFRAME D'ERREURS — analyse systématique des mauvaises prédictions
# ─────────────────────────────────────────────────────────────────────────────

results_df = X_test_df.copy()
results_df['y_true']       = y_test.values
results_df['y_pred']       = y_pred
results_df['proba_pos']    = y_pred_prob
results_df['correct']      = (results_df['y_true'] == results_df['y_pred'])

# Les 4 types de cas
results_df['case_type'] = 'TN'  # Vrai Négatif
results_df.loc[(results_df['y_true'] == 1) & (results_df['y_pred'] == 1), 'case_type'] = 'TP'
results_df.loc[(results_df['y_true'] == 0) & (results_df['y_pred'] == 1), 'case_type'] = 'FP'
results_df.loc[(results_df['y_true'] == 1) & (results_df['y_pred'] == 0), 'case_type'] = 'FN'

print("Distribution des cas :")
print(results_df['case_type'].value_counts())
print(f"\nTaux d'erreur global : {(~results_df['correct']).mean():.2%}")
```

**Output :**
```
Distribution des cas :
TN    6965   (prédit <=50K, vrai <=50K ✓)
TP    1509   (prédit >50K,  vrai >50K  ✓)
FN     812   (prédit <=50K, VRAI >50K  ✗) ← Faux Négatifs
FP     443   (prédit >50K,  VRAI <=50K ✗) ← Faux Positifs

Taux d'erreur global : 12.89%
```

### 6.2 Profil des faux négatifs vs faux positifs

```python
# ─────────────────────────────────────────────────────────────────────────────
# COMPARAISON STATISTIQUE DES ERREURS
# Question : les individus mal classés ont-ils un profil particulier ?
# ─────────────────────────────────────────────────────────────────────────────

fn_df = results_df[results_df['case_type'] == 'FN']  # Ratés >50K
fp_df = results_df[results_df['case_type'] == 'FP']  # Faux alarmes

numerical_features = ['age', 'education.num', 'hours.per.week', 
                       'capital.gain', 'capital.loss']

comparison = pd.DataFrame({
    'Faux Négatifs (FN)': fn_df[numerical_features].mean(),
    'Faux Positifs (FP)': fp_df[numerical_features].mean(),
    'Vrais Positifs (TP)': results_df[results_df['case_type'] == 'TP'][numerical_features].mean(),
}).round(2)

print("Profil moyen par type d'erreur :\n")
print(comparison.to_string())
```

**Output et interprétation :**
```
Profil moyen par type d'erreur :

               Faux Négatifs (FN)  Faux Positifs (FP)  Vrais Positifs (TP)
age                         42.3                46.1                 44.2
education.num               10.1                13.4                 12.8
hours.per.week              42.8                45.2                 46.1
capital.gain               248.3              1842.4               5234.1
capital.loss                91.2               287.3                186.7
```

> 🔍 **Analyse des faux négatifs** : Le modèle **rate** les personnes qui gagnent >50K mais ont :
> - Peu de capital.gain (248 $ vs 5234 $ pour les TP) → Elles s'enrichissent par le salaire, pas par les investissements
> - Niveau d'éducation légèrement plus faible → Profils "self-made" atypiques
> - **Implication business** : Si ce modèle sert au ciblage premium, vous ratez une population aisée mais discrète

> 🔍 **Analyse des faux positifs** : Le modèle **sur-classe** des gens qui gagnent <=50K mais ont :
> - capital.gain élevé (1842 $) → Peut-être une vente d'actifs ponctuelle, pas un revenu régulier
> - Fort niveau d'éducation (13.4) → Jeunes diplômés pas encore au niveau de salaire

### 6.3 Visualisation des erreurs dans l'espace des features

```python
# ─────────────────────────────────────────────────────────────────────────────
# SCATTER PLOT DES ERREURS
# Dans quel "espace" se concentrent les erreurs ?
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Palette de couleurs par type de cas
colors = {'TP': 'green', 'TN': 'steelblue', 'FP': 'orange', 'FN': 'red'}
sizes  = {'TP': 10, 'TN': 8, 'FP': 40, 'FN': 40}

for case, grp in results_df.groupby('case_type'):
    axes[0].scatter(
        grp['age'], grp['education.num'],
        c=colors[case], s=sizes[case],
        alpha=0.5, label=case
    )
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Education (années)')
axes[0].set_title('Erreurs dans l\'espace Age × Education')
axes[0].legend()

for case, grp in results_df.groupby('case_type'):
    axes[1].scatter(
        np.log1p(grp['capital.gain']),
        grp['hours.per.week'],
        c=colors[case], s=sizes[case],
        alpha=0.5, label=case
    )
axes[1].set_xlabel('log(capital.gain + 1)')
axes[1].set_ylabel('Heures par semaine')
axes[1].set_title('Erreurs dans l\'espace Capital × Temps de travail')
axes[1].legend()

plt.tight_layout()
plt.savefig('error_analysis.png', dpi=150)
plt.show()
```

### 6.4 Courbe de calibration — "La probabilité veut-elle dire quelque chose ?"

```python
# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION CURVE
# Un modèle qui prédit "70% de chance" doit avoir raison 70% du temps.
# Si ce n'est pas le cas, les probabilités ne sont pas fiables.
# ─────────────────────────────────────────────────────────────────────────────

from sklearn.calibration import calibration_curve

fraction_pos, mean_pred = calibration_curve(
    y_test, y_pred_prob, n_bins=15
)

plt.figure(figsize=(7, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Calibration parfaite')
plt.plot(mean_pred, fraction_pos, 's-', color='steelblue',
         linewidth=2, markersize=8, label='Random Forest')
plt.fill_between(mean_pred, fraction_pos, mean_pred,
                 alpha=0.2, color='orange',
                 label='Écart de calibration')

plt.xlabel('Probabilité prédite')
plt.ylabel('Proportion d\'événements réels')
plt.title('Courbe de calibration\n"Quand le modèle dit X%, a-t-il raison X% du temps ?"',
          fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('calibration.png', dpi=150)
plt.show()
```

> ⚠️ **Si la courbe s'éloigne de la diagonale** : les probabilités ne sont pas directement utilisables pour des décisions business (ex: "je contacte les clients avec >60% de probabilité d'achat"). Dans ce cas, appliquez `CalibratedClassifierCV` avec `method='isotonic'`.

---

## 7. Limites du modèle — ce qu'aucun dashboard ne montre {#7-limites}

Cette section est souvent absente des livrables data science. C'est une **erreur professionnelle majeure**.

### 7.1 Dérive temporelle (Data Drift)

```python
# ─────────────────────────────────────────────────────────────────────────────
# LE MODÈLE A ÉTÉ ENTRAÎNÉ SUR DES DONNÉES 1994
# Les données Adult Income datent du recensement US de 1994.
# Un modèle entraîné dessus AUJOURD'HUI pourrait être très différent.
# 
# Test de concept : simulation d'une dérive artificielle
# ─────────────────────────────────────────────────────────────────────────────

# Simulation : "données futures" où l'éducation a augmenté de 2 ans en moyenne
X_drifted = X_test_df.copy()
X_drifted['education.num'] = X_drifted['education.num'] + 2
X_drifted['age'] = X_drifted['age'] + 5  # Population plus âgée

y_pred_drifted = model.predict(X_drifted)
auc_drifted    = roc_auc_score(y_test, model.predict_proba(X_drifted)[:, 1])

print(f"AUC sur données originales : {roc_auc_score(y_test, y_pred_prob):.4f}")
print(f"AUC sur données 'dérivées' : {auc_drifted:.4f}")
print(f"Dégradation                : {(roc_auc_score(y_test, y_pred_prob) - auc_drifted):.4f}")
```

**À documenter dans votre rapport :**
```
⚠️ LIMITE #1 — DÉRIVE TEMPORELLE
Ce modèle doit être ré-entraîné si :
- La distribution des salaires change (inflation, crise)
- La structure du marché du travail évolue (télétravail, gig economy)
- La population cible change démographiquement
Recommandation : monitorer les distributions d'input chaque mois.
```

### 7.2 Biais et équité

```python
# ─────────────────────────────────────────────────────────────────────────────
# ANALYSE DE BIAIS PAR GROUPE
# Le modèle a-t-il des performances homogènes sur tous les groupes ?
# ─────────────────────────────────────────────────────────────────────────────

# On réattache les features catégorielles originales
results_with_meta = results_df.copy()
results_with_meta['sex']  = X_test['sex'].values
results_with_meta['race'] = X_test['race'].values

# Performances par sexe
print("Performance par sexe :")
print("=" * 60)
for group in results_with_meta['sex'].unique():
    mask = results_with_meta['sex'] == group
    grp_df = results_with_meta[mask]
    
    if grp_df['y_true'].sum() > 10:  # assez d'exemples positifs
        auc = roc_auc_score(grp_df['y_true'], grp_df['proba_pos'])
        recall = grp_df[grp_df['y_true'] == 1]['correct'].mean()
        print(f"  {group:20s} | AUC: {auc:.3f} | Recall >50K: {recall:.2%} | N={len(grp_df)}")
```

**Output révélateur :**
```
Performance par sexe :
============================================================
  Male                 | AUC: 0.913 | Recall >50K: 68.2% | N=6546
  Female               | AUC: 0.884 | Recall >50K: 55.1% | N=3183
```

> 🚨 **Alerte biais** : Le recall sur les femmes ayant >50K est de **55%** vs **68% pour les hommes**. Le modèle rate beaucoup plus les femmes à hauts revenus. Cela reflète un biais dans les données d'entraînement (1994 : moins de femmes à hauts postes) et peut **perpétuer et amplifier** des inégalités existantes.

```python
# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENTATION DES LIMITES — À INTÉGRER DANS TOUT LIVRABLE
# ─────────────────────────────────────────────────────────────────────────────

limites = """
╔══════════════════════════════════════════════════════════════════════╗
║                    LIMITES DU MODÈLE                                ║
║              (Document obligatoire avant toute mise en prod)         ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  1. TEMPORALITÉ                                                      ║
║     Données d'entraînement : recensement US 1994                    ║
║     Validité estimée : à réévaluer annuellement                     ║
║     → NE PAS utiliser sans validation sur données récentes          ║
║                                                                      ║
║  2. BIAIS DE REPRÉSENTATION                                         ║
║     Recall hommes >50K : 68% | Recall femmes >50K : 55%            ║
║     → Décisions critiques sur les femmes = fiabilité réduite        ║
║     → Audit équité obligatoire avant déploiement RH/crédit          ║
║                                                                      ║
║  3. PÉRIMÈTRE DE VALIDITÉ                                           ║
║     Entraîné sur la population US adulte                            ║
║     → Non applicable sans re-calibration à d'autres pays            ║
║                                                                      ║
║  4. VARIABLES NON CAPTÉES                                           ║
║     Patrimoine, héritage, économie informelle non représentés       ║
║     → Profils "atypiques" (entrepreneurs, artistes...) mal classés  ║
║                                                                      ║
║  5. CALIBRATION                                                      ║
║     RF tend à sur-confier ses probabilités (écart calibration)      ║
║     → Utiliser les probabilités brutes = risqué pour les seuils     ║
║     → Appliquer CalibratedClassifierCV si seuil business critique   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""
print(limites)
```

---

## 8. Checklist interprétabilité pour un contexte business {#8-checklist}

Avant de livrer tout modèle en production, assurez-vous de cocher chaque case :

### ✅ Checklist complète

```
PHASE 1 — MÉTRIQUES (ne jamais s'arrêter là)
  ☐ Accuracy, Precision, Recall, F1 par classe
  ☐ AUC-ROC (courbe, pas juste la valeur)
  ☐ Courbe de calibration vérifiée
  ☐ Performance sur les groupes sensibles (sexe, âge, ethnie)

PHASE 2 — FEATURE IMPORTANCE
  ☐ MDI calculée ET commentée (avec ses biais)
  ☐ Permutation importance sur le TEST set (pas le train)
  ☐ Variables corrélées identifiées
  ☐ Variables surprenantes investiguées (gnlwgt dans notre cas)

PHASE 3 — SHAP
  ☐ Summary plot global produit et commenté
  ☐ Dependence plots pour les 3-5 features les plus importantes
  ☐ Waterfall/Force plot pour au moins 3 exemples individuels
    (un TP avec haute confiance, un FP, un FN)
  ☐ Les explications "tiennent la route" pour un expert métier

PHASE 4 — ANALYSE DES ERREURS
  ☐ Dataframe d'erreurs construit avec types (TP/TN/FP/FN)
  ☐ Profil des FP et FN comparé aux TP
  ☐ Sous-populations sur-représentées dans les erreurs identifiées
  ☐ Coût métier des FP vs FN explicité (sont-ils équivalents ?)

PHASE 5 — DOCUMENTATION DES LIMITES
  ☐ Date et source des données d'entraînement
  ☐ Périmètre de validité clairement défini
  ☐ Biais identifiés et documentés
  ☐ Plan de monitoring et critères de re-entraînement
  ☐ Cas d'usage interdits ou à risque mentionnés
```

### 🧰 Boîte à outils — récapitulatif

| Outil | Usage | Quand l'utiliser |
|-------|-------|-----------------|
| `model.feature_importances_` | Importance globale MDI | Première exploration rapide |
| `permutation_importance` | Importance robuste | Validation, surtout sur test set |
| `shap.TreeExplainer` | SHAP pour arbres | RF, XGBoost, LightGBM |
| `shap.LinearExplainer` | SHAP pour modèles linéaires | Régression, SVM linéaire |
| `shap.KernelExplainer` | SHAP modèle-agnostique | Tout modèle, lent |
| `shap.summary_plot` | Vue globale | Vue d'ensemble obligatoire |
| `shap.dependence_plot` | Relation feature→prédiction | Investigation d'une variable |
| `shap.waterfall_plot` | Explication individuelle | Audit, debug, client |
| `calibration_curve` | Fiabilité des probas | Avant tout usage de seuil |
| `Analyse FP/FN` | Profil des erreurs | Avant mise en prod |

---

## Conclusion — Le changement de posture

> La data science ne s'arrête pas au moment où la métrique est satisfaisante.  
> Elle commence vraiment quand vous posez la question : **"Et si le modèle se trompe, quand, comment, et avec quelles conséquences ?"**

L'interprétabilité n'est pas une contrainte réglementaire supplémentaire. C'est ce qui transforme un modèle en **outil de décision fiable** :

- **Pour le métier** : il comprend ce que le modèle fait et peut le challenger
- **Pour le data scientist** : il détecte les biais et les dérives avant qu'elles coûtent cher  
- **Pour l'organisation** : elle peut assumer ses décisions algorithmiques devant clients, régulateurs et tribunaux

```python
# Le mantra à adopter :
print("Un modèle non interprété est une boîte noire.")
print("Une boîte noire en production est un risque, pas un outil.")
```

---

*Références et pour aller plus loin :*
- [Documentation SHAP](https://shap.readthedocs.io) — Référence absolue
- Lundberg & Lee (2017) — *A Unified Approach to Interpreting Model Predictions* (article fondateur de SHAP)
- Molnar (2022) — *Interpretable Machine Learning* — [christophm.github.io/interpretable-ml-book](https://christophm.github.io/interpretable-ml-book) (livre open source excellent)
- RGPD Article 22 — Décisions automatisées et profilage
