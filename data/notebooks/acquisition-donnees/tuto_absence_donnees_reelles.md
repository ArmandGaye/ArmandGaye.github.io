# 🧪 Travailler sans données réelles — Guide complet
## Trois cas concrets, trois stratégies, zéro improvisation

> **Problème central** : En data science, on dispose rarement des données "idéales" au bon moment.  
> Ce tutoriel répond à une question que beaucoup évitent : **comment travailler sérieusement quand les vraies données n'existent pas encore (ou ne peuvent pas être partagées) ?**

---

## Table des matières

1. [Philosophie générale — pourquoi simuler des données n'est pas tricher](#0-philosophie)
2. [CAS 1 — Données scrapées par vous-même](#cas1)
3. [CAS 2 — Dataset métier peu connu ou confidentiel](#cas2)
4. [CAS 3 — Données combinées multi-sources (API + Kaggle + enrichissement)](#cas3)
5. [Validation croisée — comment savoir si votre simulation est crédible](#validation)
6. [Checklist avant de passer aux vraies données](#checklist)

---

## 1. Philosophie générale {#0-philosophie}

### Simuler des données n'est pas tricher — c'est une discipline

Il existe une confusion fréquente entre :

| Pratique | Statut | Objectif |
|----------|--------|----------|
| **Inventer des données** et prétendre qu'elles sont réelles | ❌ Fraude | Tromper |
| **Simuler des données** avec des hypothèses explicites | ✅ Méthode scientifique | Tester, prototyper, documenter |
| **Générer des données synthétiques** à partir d'une vraie distribution | ✅ Pratique standard | Anonymiser, augmenter |

La simulation de données est utilisée quotidiennement par :
- Les statisticiens (Monte Carlo, bootstrapping)
- Les ingénieurs (tests A/B avant déploiement)
- Les data scientists (prototypage de pipelines)
- Les chercheurs (reproducibility, peer review)

### La règle d'or : **toujours documenter vos hypothèses**

```python
# ❌ MAUVAIS — hypothèse cachée
ages = np.random.normal(35, 10, 1000)

# ✅ BON — hypothèse explicite et traçable
# Hypothèse : âge moyen des clients = 35 ans, std = 10 ans
# Source : rapport sectoriel XYZ 2023, page 14
# À réviser dès accès aux vraies données
ages = np.random.normal(loc=35, scale=10, size=1000)
ages = np.clip(ages, 18, 80)  # Contrainte métier : clients majeurs, max 80 ans
```

---

## CAS 1 — Données scrapées par vous-même {#cas1}

### Contexte

Vous voulez analyser les annonces immobilières d'un site web pour construire un modèle de prix. Le problème : le scraping prend du temps, les sites bloquent les robots, les données changent chaque jour, et vous avez besoin de travailler **maintenant** pour valider votre pipeline.

### Stratégie

```
Étape 1 : Scraper un petit échantillon réel (50-100 lignes)
           → Comprendre la vraie structure des données
           
Étape 2 : Analyser les distributions du mini-échantillon
           → Extraire les statistiques clés (moyenne, std, corrélations)
           
Étape 3 : Générer un dataset synthétique qui respecte ces distributions
           → Travailler à grande échelle
           
Étape 4 : Documenter les hypothèses et les écarts connus
           → Ne jamais oublier que ce sont des simulations
```

---

### 1.1 Simulation d'un scraper d'annonces immobilières

```python
import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

# pip install faker

fake = Faker('fr_FR')
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 1 : DÉFINIR LA STRUCTURE RÉELLE (comme si on avait scrapé 50 annonces)
#
# Ces statistiques "viennent" de notre mini-échantillon réel.
# En pratique : faire un vrai scraping de 50 lignes, calculer ces stats.
# ─────────────────────────────────────────────────────────────────────────────

# Structure observée sur le site (exemple SeLoger / LeBonCoin Immo)
VILLES = {
    'Paris 15e':       {'prix_m2_moy': 10200, 'prix_m2_std': 1200, 'poids': 0.15},
    'Paris 11e':       {'prix_m2_moy':  9800, 'prix_m2_std': 1100, 'poids': 0.12},
    'Lyon 3e':         {'prix_m2_moy':  4800, 'prix_m2_std':  700, 'poids': 0.10},
    'Lyon 6e':         {'prix_m2_moy':  5500, 'prix_m2_std':  900, 'poids': 0.08},
    'Bordeaux Centre': {'prix_m2_moy':  5100, 'prix_m2_std':  800, 'poids': 0.09},
    'Marseille 6e':    {'prix_m2_moy':  4200, 'prix_m2_std':  600, 'poids': 0.08},
    'Nantes Centre':   {'prix_m2_moy':  4600, 'prix_m2_std':  750, 'poids': 0.07},
    'Toulouse':        {'prix_m2_moy':  4100, 'prix_m2_std':  650, 'poids': 0.10},
    'Strasbourg':      {'prix_m2_moy':  3800, 'prix_m2_std':  600, 'poids': 0.08},
    'Banlieue Paris':  {'prix_m2_moy':  5800, 'prix_m2_std':  900, 'poids': 0.13},
}

TYPE_BIEN = ['Appartement', 'Maison', 'Studio', 'Loft']

def generer_annonce(id_annonce):
    """
    Génère une annonce immobilière synthétique.
    Chaque paramètre est documenté avec son hypothèse source.
    """
    # Sélection de la ville selon les poids observés
    ville = random.choices(
        list(VILLES.keys()),
        weights=[v['poids'] for v in VILLES.values()]
    )[0]
    
    config = VILLES[ville]
    
    # Type de bien (distribution observée dans le mini-échantillon)
    type_bien = random.choices(
        TYPE_BIEN,
        weights=[0.55, 0.25, 0.15, 0.05]  # 55% apparts, 25% maisons...
    )[0]
    
    # Surface (distribution log-normale observée — asymétrie vers petites surfaces)
    # Hypothèse : médiane 65m², log-std de 0.35
    if type_bien == 'Studio':
        surface = max(12, np.random.lognormal(mean=np.log(28), sigma=0.25))
    elif type_bien == 'Maison':
        surface = max(50, np.random.lognormal(mean=np.log(110), sigma=0.3))
    elif type_bien == 'Loft':
        surface = max(40, np.random.lognormal(mean=np.log(80), sigma=0.35))
    else:  # Appartement
        surface = max(18, np.random.lognormal(mean=np.log(65), sigma=0.35))
    surface = round(surface, 1)
    
    # Prix au m² (lié à la ville, avec un peu de bruit)
    prix_m2 = max(1500, np.random.normal(config['prix_m2_moy'], config['prix_m2_std']))
    
    # Effet surface sur le prix/m² (décroissance légère pour les grands biens)
    # Observation empirique : +10m² → -1% sur le prix/m²
    facteur_surface = 1 - 0.001 * max(0, surface - 50)
    prix_m2 *= facteur_surface
    
    # Nombre de pièces
    pieces = max(1, int(surface / 20) + random.randint(-1, 1))
    pieces = min(pieces, 8)
    
    # Étage (pour les appartements)
    etage = None
    has_ascenseur = False
    if type_bien in ['Appartement', 'Studio', 'Loft']:
        etage = random.randint(0, 8)
        has_ascenseur = (etage >= 3) or (random.random() < 0.4)
        # Bonus étage élevé avec ascenseur
        if etage >= 4 and has_ascenseur:
            prix_m2 *= 1.05
    
    # Caractéristiques booléennes
    has_parking    = random.random() < 0.35  # 35% des annonces ont un parking
    has_balcon     = random.random() < 0.40
    has_cave       = random.random() < 0.50
    has_gardien    = random.random() < 0.15 if 'Paris' in ville else random.random() < 0.05
    
    # DPE (Diagnostic de Performance Énergétique)
    # Distribution observée : majorité E-F dans les grandes villes
    dpe = random.choices(
        ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        weights=[0.03, 0.07, 0.12, 0.18, 0.25, 0.22, 0.13]
    )[0]
    
    # Effet DPE sur le prix (décote pour mauvais DPE, observée sur le marché)
    dpe_malus = {'A': 1.08, 'B': 1.04, 'C': 1.01, 'D': 1.0,
                 'E': 0.97, 'F': 0.93, 'G': 0.88}
    prix_m2 *= dpe_malus[dpe]
    
    # Prix total
    prix_total = round(prix_m2 * surface, -2)  # Arrondi à la centaine
    
    # Date de publication
    jours_ago = int(np.random.exponential(scale=15))  # La plupart récentes
    jours_ago = min(jours_ago, 180)
    date_pub = datetime.now() - timedelta(days=jours_ago)
    
    # Titre d'annonce (simulé avec Faker)
    titres = [
        f"{type_bien} {pieces}P - {round(surface)}m² - {ville}",
        f"Beau {type_bien.lower()} {round(surface)}m² {ville}",
        f"{type_bien} {round(surface)}m² {pieces} pièces",
    ]
    
    return {
        'id':              f"IMM-{id_annonce:06d}",
        'titre':           random.choice(titres),
        'ville':           ville,
        'type_bien':       type_bien,
        'surface_m2':      surface,
        'nb_pieces':       pieces,
        'etage':           etage,
        'has_ascenseur':   has_ascenseur,
        'has_parking':     has_parking,
        'has_balcon':      has_balcon,
        'has_cave':        has_cave,
        'has_gardien':     has_gardien,
        'dpe':             dpe,
        'prix_total':      prix_total,
        'prix_m2':         round(prix_m2, 0),
        'date_publication': date_pub.strftime('%Y-%m-%d'),
        'nb_photos':       random.randint(0, 15),
        # Simulation du nombre de vues (corrélé au prix et à la date)
        'nb_vues':         max(5, int(
            np.random.normal(200, 80) * (1 + jours_ago * 0.05)
        )),
    }

# ─────────────────────────────────────────────────────────────────────────────
# GÉNÉRATION DU DATASET
# ─────────────────────────────────────────────────────────────────────────────

n_annonces = 5000
annonces = [generer_annonce(i) for i in range(1, n_annonces + 1)]
df_immo = pd.DataFrame(annonces)

print(f"Dataset généré : {df_immo.shape}")
print(f"\nAperçu :")
print(df_immo.head(3).to_string())
print(f"\nStatistiques prix :")
print(df_immo['prix_m2'].describe().round(0))
```

---

### 1.2 Validation de la cohérence du dataset simulé

```python
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────────────────────────
# TEST DE COHÉRENCE — Le dataset simulé est-il crédible ?
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Validation du dataset immobilier simulé', fontsize=14, fontweight='bold')

# 1. Distribution des prix au m² par ville (doit ressembler à la réalité)
axes[0, 0].set_title('Distribution prix/m² par ville')
city_order = df_immo.groupby('ville')['prix_m2'].median().sort_values(ascending=False).index
df_immo.boxplot(column='prix_m2', by='ville', ax=axes[0, 0], rot=45)

# 2. Relation surface ↔ prix total (doit être croissante et linéaire)
axes[0, 1].set_title('Surface vs Prix total')
axes[0, 1].scatter(df_immo['surface_m2'], df_immo['prix_total'],
                   alpha=0.1, s=5, c='steelblue')
axes[0, 1].set_xlabel('Surface (m²)')
axes[0, 1].set_ylabel('Prix total (€)')

# Ligne de tendance
z = np.polyfit(df_immo['surface_m2'], df_immo['prix_total'], 1)
p = np.poly1d(z)
x_line = np.linspace(10, 300, 100)
axes[0, 1].plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Tendance')
axes[0, 1].legend()

# 3. Distribution DPE (doit ressembler aux stats INSEE)
axes[0, 2].set_title('Distribution DPE (vs INSEE réel)')
dpe_counts = df_immo['dpe'].value_counts().sort_index()
dpe_colors = {'A':'#1a9850','B':'#91cf60','C':'#d9ef8b',
              'D':'#fee08b','E':'#fc8d59','F':'#d73027','G':'#a50026'}
axes[0, 2].bar(dpe_counts.index,
               dpe_counts.values / len(df_immo) * 100,
               color=[dpe_colors[d] for d in dpe_counts.index])
axes[0, 2].set_ylabel('% des annonces')
axes[0, 2].set_xlabel('Classe DPE')

# 4. Distribution des surfaces (doit être log-normale)
axes[1, 0].set_title('Distribution des surfaces')
axes[1, 0].hist(df_immo['surface_m2'], bins=50, color='steelblue', alpha=0.7, edgecolor='white')
axes[1, 0].set_xlabel('Surface (m²)')
axes[1, 0].axvline(df_immo['surface_m2'].median(), color='red',
                   linestyle='--', label=f"Médiane : {df_immo['surface_m2'].median():.0f}m²")
axes[1, 0].legend()

# 5. Effet DPE sur le prix (doit montrer une décote visible)
axes[1, 1].set_title('Prix moyen par DPE (décote attendue)')
prix_dpe = df_immo.groupby('dpe')['prix_m2'].mean().sort_index()
axes[1, 1].bar(prix_dpe.index, prix_dpe.values,
               color=[dpe_colors[d] for d in prix_dpe.index])
axes[1, 1].set_ylabel('Prix moyen (€/m²)')
axes[1, 1].set_xlabel('Classe DPE')

# 6. Répartition par type de bien
axes[1, 2].set_title('Répartition par type de bien')
type_counts = df_immo['type_bien'].value_counts()
axes[1, 2].pie(type_counts.values, labels=type_counts.index,
               autopct='%1.1f%%', startangle=90,
               colors=['#4878CF','#6ACC65','#D65F5F','#B47CC7'])

plt.tight_layout()
plt.savefig('validation_immo_simule.png', dpi=150)
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# RAPPORT DE COHÉRENCE AUTOMATIQUE
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("RAPPORT DE COHÉRENCE — Dataset Immobilier Simulé")
print("="*60)

checks = [
    ("Prix/m² Paris > Prix/m² Lyon",
     df_immo[df_immo['ville'].str.contains('Paris')]['prix_m2'].mean() >
     df_immo[df_immo['ville'].str.contains('Lyon')]['prix_m2'].mean()),
    
    ("Surface médiane dans [40, 90] m²",
     40 < df_immo['surface_m2'].median() < 90),
    
    ("Prix total corrélé positivement à la surface",
     df_immo['surface_m2'].corr(df_immo['prix_total']) > 0.7),
    
    ("DPE E+F+G représentent > 50% (parc immobilier français)",
     df_immo['dpe'].isin(['E','F','G']).mean() > 0.5),
    
    ("% biens avec parking < 50%",
     df_immo['has_parking'].mean() < 0.5),
    
    ("Pas de prix négatifs ou nuls",
     (df_immo['prix_total'] > 0).all()),
]

for desc, result in checks:
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"  {status} — {desc}")
```

---

### 1.3 Pipeline de remplacement : prêt pour les vraies données

```python
# ─────────────────────────────────────────────────────────────────────────────
# PATTERN RECOMMANDÉ : data_loader() abstrait la source
# Quand les vraies données arrivent, SEULE cette fonction change.
# Tout le reste du pipeline reste intact.
# ─────────────────────────────────────────────────────────────────────────────

def load_data(source='simulated', filepath=None, n_samples=5000):
    """
    Charge les données depuis la source spécifiée.
    
    Args:
        source: 'simulated' | 'csv' | 'database' | 'scraper'
        filepath: chemin vers le CSV si source='csv'
        n_samples: nombre de lignes si source='simulated'
    
    Returns:
        pd.DataFrame avec les colonnes standardisées
    """
    if source == 'simulated':
        print("⚠️  DONNÉES SIMULÉES — Ne pas utiliser pour des décisions finales")
        data = [generer_annonce(i) for i in range(n_samples)]
        df = pd.DataFrame(data)
        
    elif source == 'csv':
        df = pd.read_csv(filepath)
        # Appliquer les mêmes normalisations
        
    elif source == 'database':
        # df = pd.read_sql("SELECT ...", connection)
        raise NotImplementedError("Connexion DB pas encore configurée")
        
    elif source == 'scraper':
        # Importer le vrai scraper quand prêt
        # from scrapers.seloger import scrape_annonces
        # df = scrape_annonces(pages=50)
        raise NotImplementedError("Scraper en cours de développement")
    
    # Validation du schéma (identique quelle que soit la source)
    colonnes_requises = ['surface_m2', 'prix_total', 'ville', 'type_bien']
    for col in colonnes_requises:
        assert col in df.columns, f"Colonne manquante : {col}"
    
    return df

# Usage
df = load_data(source='simulated', n_samples=5000)
# Plus tard : df = load_data(source='csv', filepath='data/annonces_mai2025.csv')
# Plus tard : df = load_data(source='database')
```

> 💡 **Bonne pratique** : Ce pattern `load_data()` est la différence entre un prototype qui devient production et un prototype qui se réécrit de zéro.

---

## CAS 2 — Dataset métier peu connu ou confidentiel {#cas2}

### Contexte

Vous êtes consultant pour une clinique. Ils veulent prédire les no-shows (patients qui ne se présentent pas). Leurs données sont confidentielles (données médicales = RGPD), pas encore disponibles, et vous devez présenter une preuve de concept dans 10 jours.

### Stratégie

```
1. Documenter la structure attendue à partir d'entretiens métier
2. Trouver des études publiées sur le phénomène (taux de no-show, corrélats)
3. Générer un dataset synthétique réaliste avec les bonnes dépendances
4. Ajouter les imperfections réalistes (valeurs manquantes, erreurs de saisie)
5. Tester le pipeline complet avant d'avoir une seule ligne réelle
```

---

### 2.1 Dataset médical synthétique : no-show de consultations

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# ─────────────────────────────────────────────────────────────────────────────
# CONNAISSANCES MÉTIER ENCODÉES
# Sources : 
#   - Entretiens avec le responsable médical (durée des créneaux, spécialités)
#   - Publication : "Factors associated with no-show", BMC Health Services (2019)
#   - Rapport interne anonymisé 2022 : taux global no-show = 18% dans cette clinique
# ─────────────────────────────────────────────────────────────────────────────

SPECIALITES = {
    'Médecine générale':      {'duree_min': 15, 'taux_no_show_base': 0.22},
    'Cardiologie':            {'duree_min': 30, 'taux_no_show_base': 0.12},
    'Dermatologie':           {'duree_min': 20, 'taux_no_show_base': 0.25},
    'Gynécologie':            {'duree_min': 25, 'taux_no_show_base': 0.14},
    'Pédiatrie':              {'duree_min': 20, 'taux_no_show_base': 0.18},
    'Psychiatrie':            {'duree_min': 45, 'taux_no_show_base': 0.35},  # Plus élevé (littérature)
    'Ophtalmologie':          {'duree_min': 20, 'taux_no_show_base': 0.16},
    'Orthopédie':             {'duree_min': 30, 'taux_no_show_base': 0.13},
}

JOURS_SEMAINE = {0: 'Lundi', 1: 'Mardi', 2: 'Mercredi', 3: 'Jeudi', 4: 'Vendredi'}

np.random.seed(42)

def calculer_proba_no_show(row_data):
    """
    Calcule la probabilité de no-show basée sur les facteurs connus.
    Chaque facteur est justifié par la littérature ou les entretiens.
    """
    spec = SPECIALITES[row_data['specialite']]
    p = spec['taux_no_show_base']
    
    # Facteur 1 : délai de prise de RDV
    # Littérature : chaque semaine supplémentaire → +2% no-show (approx.)
    delai_semaines = row_data['delai_rdv_jours'] / 7
    p += 0.02 * min(delai_semaines, 8)  # Plafonné à 8 semaines
    
    # Facteur 2 : heure du rendez-vous
    # Observation interne : créneaux 8h00 et 17h00+ ont plus de no-shows
    heure = row_data['heure_rdv']
    if heure <= 8:
        p += 0.05  # Tôt le matin
    elif heure >= 17:
        p += 0.08  # Fin de journée
    elif 12 <= heure <= 14:
        p += 0.03  # Heure du déjeuner
    
    # Facteur 3 : jour de la semaine
    jour = row_data['jour_semaine']
    if jour == 'Lundi':
        p += 0.03  # Début de semaine : taux légèrement supérieur
    elif jour == 'Vendredi':
        p += 0.05  # Veille de week-end
    
    # Facteur 4 : patient fidèle vs nouveau patient
    if row_data['nb_rdv_precedents'] == 0:
        p += 0.08  # Nouveau patient : plus de no-show
    elif row_data['nb_rdv_precedents'] >= 5:
        p -= 0.05  # Patient régulier : moins de no-show
    
    # Facteur 5 : historique de no-show du patient
    if row_data['nb_no_show_precedents'] >= 2:
        p += 0.20  # Fort prédicteur dans la littérature
    elif row_data['nb_no_show_precedents'] == 1:
        p += 0.08
    
    # Facteur 6 : confirmation SMS/email (si disponible)
    if row_data['confirmation_envoyee']:
        p -= 0.07  # Réduction observée en interne
    
    # Facteur 7 : type de consultation
    if row_data['type_consultation'] == 'Suivi':
        p -= 0.03  # Les suivis ont légèrement moins de no-show
    elif row_data['type_consultation'] == 'Urgence':
        p -= 0.10  # Les urgences : très peu de no-show
    
    # Facteur 8 : âge du patient
    age = row_data['age_patient']
    if 18 <= age <= 30:
        p += 0.05   # Jeunes adultes : plus de no-show
    elif age >= 65:
        p -= 0.03   # Seniors : plus assidus
    
    return np.clip(p, 0.02, 0.95)  # Bornes réalistes

def generer_rdv(id_rdv):
    """Génère un rendez-vous médical synthétique avec toutes ses caractéristiques."""
    
    # Date de RDV (sur 12 mois glissants)
    date_rdv = datetime.now() - timedelta(days=random.randint(0, 365))
    while date_rdv.weekday() >= 5:  # Pas le week-end
        date_rdv = date_rdv - timedelta(days=1)
    
    specialite = random.choice(list(SPECIALITES.keys()))
    
    # Heure selon distribution horaire réaliste (pic 9h-11h et 14h-16h)
    heures_disponibles = list(range(8, 19))
    poids_heures = [0.05,0.12,0.15,0.13,0.06,0.04,0.11,0.14,0.12,0.05,0.03]
    heure_rdv = random.choices(heures_disponibles, weights=poids_heures)[0]
    
    # Patient
    age = int(np.clip(np.random.normal(45, 18), 1, 95))
    nb_rdv_prec = int(np.random.exponential(scale=3))
    nb_no_show_prec = min(nb_rdv_prec, int(np.random.exponential(scale=0.4)))
    
    # Délai de prise de RDV (log-normal, médiane ~14 jours)
    delai = max(0, int(np.random.lognormal(mean=np.log(14), sigma=0.8)))
    
    row = {
        'id_rdv':                  f"RDV-{id_rdv:07d}",
        'date_rdv':                date_rdv.strftime('%Y-%m-%d'),
        'heure_rdv':               heure_rdv,
        'jour_semaine':            JOURS_SEMAINE.get(date_rdv.weekday(), 'Vendredi'),
        'specialite':              specialite,
        'duree_consultation_min':  SPECIALITES[specialite]['duree_min'],
        'type_consultation':       random.choices(
                                       ['Première consultation', 'Suivi', 'Urgence', 'Bilan'],
                                       weights=[0.30, 0.45, 0.10, 0.15]
                                   )[0],
        'age_patient':             age,
        'sexe_patient':            random.choice(['M', 'F']),
        'code_postal':             fake.postcode() if hasattr(fake, 'postcode') else f"7{random.randint(1000,5000)}",
        'delai_rdv_jours':         delai,
        'nb_rdv_precedents':       nb_rdv_prec,
        'nb_no_show_precedents':   nb_no_show_prec,
        'confirmation_envoyee':    random.random() < 0.65,  # 65% reçoivent une confirmation
        'confirmation_lue':        False,  # Rempli ci-dessous
        'mode_prise_rdv':          random.choices(
                                       ['Téléphone', 'En ligne', 'En personne', 'Application'],
                                       weights=[0.45, 0.30, 0.15, 0.10]
                                   )[0],
    }
    
    # Confirmation lue si confirmation envoyée (80% des envoyées)
    if row['confirmation_envoyee']:
        row['confirmation_lue'] = random.random() < 0.80
    
    # Calcul de la probabilité de no-show réelle
    proba_no_show = calculer_proba_no_show(row)
    
    # Tirage final avec bruit aléatoire résiduel (ce que le modèle ne peut pas capter)
    row['no_show'] = int(random.random() < proba_no_show)
    row['proba_no_show_theorique'] = round(proba_no_show, 3)  # Pour validation seulement
    
    # ─────────────────────────────────────────────────────────────────────────
    # INJECTION DE RÉALISME : imperfections des données médicales
    # ─────────────────────────────────────────────────────────────────────────
    
    # 3% des âges manquants (non renseignés à la prise de RDV)
    if random.random() < 0.03:
        row['age_patient'] = np.nan
    
    # 8% des codes postaux manquants
    if random.random() < 0.08:
        row['code_postal'] = None
    
    # Erreurs de saisie : heure parfois mal saisie (1% des cas)
    if random.random() < 0.01:
        row['heure_rdv'] = random.choice([0, 99, -1])  # Valeur aberrante
    
    return row

# Génération
print("Génération du dataset médical synthétique...")
rdv_data = [generer_rdv(i) for i in range(1, 10001)]
df_medical = pd.DataFrame(rdv_data)

# Retirer la colonne de vérité terrain (ne serait pas disponible en prod)
df_medical_clean = df_medical.drop(columns=['proba_no_show_theorique'])

print(f"\nDataset généré : {df_medical_clean.shape}")
print(f"Taux de no-show global : {df_medical_clean['no_show'].mean():.1%}")
print(f"\nTaux de no-show par spécialité :")
print(df_medical_clean.groupby('specialite')['no_show'].mean()
      .sort_values(ascending=False).round(3).to_string())
```

---

### 2.2 Vérification : les patterns métier sont-ils reproduits ?

```python
# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION MÉTIER — Les patterns connus sont-ils présents ?
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*65)
print("VALIDATION DES PATTERNS MÉTIER")
print("="*65)

# Test 1 : L'historique de no-show est prédictif
ns_history = df_medical_clean.groupby('nb_no_show_precedents')['no_show'].mean()
print("\n1. No-show selon l'historique (doit croître) :")
for k, v in ns_history.head(5).items():
    bar = "█" * int(v * 30)
    print(f"   {k} no-show passés → {v:.1%} {bar}")

# Test 2 : La psychiatrie a le taux le plus élevé
max_spec = df_medical_clean.groupby('specialite')['no_show'].mean().idxmax()
print(f"\n2. Spécialité avec le plus de no-show : {max_spec}")
print(f"   ({'✅ cohérent' if max_spec == 'Psychiatrie' else '❌ inattendu'} avec la littérature)")

# Test 3 : La confirmation réduit les no-show
ns_conf = df_medical_clean.groupby('confirmation_envoyee')['no_show'].mean()
print(f"\n3. Effet confirmation SMS :")
print(f"   Sans confirmation : {ns_conf[False]:.1%}")
print(f"   Avec confirmation : {ns_conf[True]:.1%}")
reduction = ns_conf[False] - ns_conf[True]
print(f"   Réduction : {reduction:.1%} ({'✅ positif' if reduction > 0 else '❌ problème'})")

# Test 4 : Délai corrélé positivement au no-show
corr_delai = df_medical_clean['delai_rdv_jours'].corr(df_medical_clean['no_show'])
print(f"\n4. Corrélation délai ↔ no-show : {corr_delai:.3f}")
print(f"   ({'✅ positive comme attendu' if corr_delai > 0 else '❌ problème'})")

# Test 5 : % valeurs manquantes réaliste
print(f"\n5. Taux de valeurs manquantes :")
print(df_medical_clean.isnull().mean()[df_medical_clean.isnull().mean() > 0].round(3))
```

---

### 2.3 Modèle complet sur données synthétiques

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────────────────────────────────
# ENTRAÎNEMENT COMPLET — Pipeline prêt pour les vraies données
# ─────────────────────────────────────────────────────────────────────────────

# Preprocessing
df_model = df_medical_clean.copy()

# Gestion des valeurs manquantes
df_model['age_patient'] = df_model['age_patient'].fillna(df_model['age_patient'].median())
df_model['code_postal'] = df_model['code_postal'].fillna('INCONNU')

# Nettoyage des heures aberrantes
df_model = df_model[df_model['heure_rdv'].between(7, 20)]

# Encodage des variables catégorielles
cat_cols = ['specialite', 'type_consultation', 'jour_semaine',
            'mode_prise_rdv', 'sexe_patient']
le = LabelEncoder()
for col in cat_cols:
    df_model[col + '_enc'] = le.fit_transform(df_model[col].astype(str))

# Features
features = [
    'heure_rdv', 'delai_rdv_jours', 'nb_rdv_precedents',
    'nb_no_show_precedents', 'confirmation_envoyee', 'confirmation_lue',
    'age_patient', 'duree_consultation_min',
] + [col + '_enc' for col in cat_cols]

X = df_model[features]
y = df_model['no_show']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Modèle
model = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.1,
    max_depth=4, random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("RÉSULTATS SUR DONNÉES SYNTHÉTIQUES")
print("(Performances à ne pas extrapoler aux vraies données)")
print("="*55)
print(classification_report(y_test, y_pred, target_names=['Présent', 'No-show']))

# Feature importance commentée
fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\nTop 8 features prédictives :")
for feat, imp in fi.head(8).items():
    print(f"  {feat:35s} {imp:.3f}  {'█' * int(imp * 100)}")
```

---

## CAS 3 — Données combinées multi-sources {#cas3}

### Contexte

Vous construisez un système de recommandation de restaurants pour une startup. Vous avez accès à : (1) l'API OpenFoodFacts (données nutritionnelles), (2) un export Kaggle de restaurants (Zomato dataset), (3) des données de fréquentation GPS open data. Mais les API ne sont pas stables, le Kaggle nécessite du nettoyage, et vous voulez enrichir tout ça avec des données météo. Comment travailler en avance sur tout ça ?

### Stratégie

```
1. Définir le schéma cible (comment vos 3 sources se rejoignent)
2. Simuler chaque source INDÉPENDAMMENT en respectant sa structure réelle
3. Implémenter la logique de jointure sur données simulées
4. Tester la pipeline d'enrichissement de bout en bout
5. Basculer source par source vers les vraies données
```

---

### 3.1 Simulation de chaque source

```python
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════
# SOURCE A — Simulation d'une réponse API (style OpenFoodFacts / Yelp)
# Structure réelle documentée : https://world.openfoodfacts.org/data
# ═══════════════════════════════════════════════════════════════════════════

CUISINES = ['Française', 'Italienne', 'Japonaise', 'Mexicaine', 'Indienne',
            'Libanaise', 'Végétarienne', 'Américaine', 'Thaïlandaise', 'Africaine']

QUARTIERS_PARIS = {
    'Marais':        {'lat_center': 48.857, 'lon_center': 2.354, 'standing': 'haut'},
    'Montmartre':    {'lat_center': 48.886, 'lon_center': 2.342, 'standing': 'moyen'},
    'Saint-Germain': {'lat_center': 48.854, 'lon_center': 2.333, 'standing': 'haut'},
    'Belleville':    {'lat_center': 48.872, 'lon_center': 2.378, 'standing': 'bas'},
    'Bastille':      {'lat_center': 48.853, 'lon_center': 2.369, 'standing': 'moyen'},
    'Oberkampf':     {'lat_center': 48.864, 'lon_center': 2.371, 'standing': 'moyen'},
    'Batignolles':   {'lat_center': 48.885, 'lon_center': 2.319, 'standing': 'moyen'},
    'Châtelet':      {'lat_center': 48.860, 'lon_center': 2.347, 'standing': 'mixte'},
}

def simulate_api_response_restaurant(restaurant_id):
    """
    Simule une réponse d'API restaurant (style Yelp/Google Places).
    Chaque champ correspond à un vrai champ de l'API documentée.
    """
    quartier = random.choice(list(QUARTIERS_PARIS.keys()))
    config   = QUARTIERS_PARIS[quartier]
    standing = config['standing']
    
    # Prix lié au standing du quartier
    fourchette_prix = {
        'haut': random.choice(['€€€', '€€€€']),
        'moyen': random.choice(['€€', '€€€']),
        'bas': random.choice(['€', '€€']),
        'mixte': random.choice(['€', '€€', '€€€']),
    }[standing]
    
    # Note Yelp-like : distribution bêta (biais positif, mais réaliste)
    note = round(np.random.beta(a=8, b=2) * 4 + 1, 1)  # Entre 1.0 et 5.0
    note = min(5.0, max(1.0, note))
    
    # Nombre d'avis (log-normal)
    nb_avis = max(5, int(np.random.lognormal(mean=np.log(150), sigma=1.2)))
    
    cuisine = random.choice(CUISINES)
    
    return {
        # Champs standard de l'API
        "id":          f"rest_{restaurant_id:06d}",
        "name":        f"Restaurant {fake.last_name()} {random.choice(['& Co', 'Bistro', 'Café', 'Brasserie', ''])}",
        "cuisine":     cuisine,
        "address": {
            "street":     fake.street_address(),
            "quartier":   quartier,
            "city":       "Paris",
            "zip_code":   f"750{random.randint(1,20):02d}",
        },
        "location": {
            "lat": config['lat_center'] + np.random.normal(0, 0.008),
            "lon": config['lon_center'] + np.random.normal(0, 0.010),
        },
        "rating":       note,
        "review_count": nb_avis,
        "price_range":  fourchette_prix,
        "open_now":     random.random() < 0.65,
        "hours":        {
            "monday":    "12:00-14:30,19:00-22:30",
            "tuesday":   "12:00-14:30,19:00-22:30",
            "wednesday": "12:00-14:30,19:00-22:30",
            "thursday":  "12:00-14:30,19:00-23:00",
            "friday":    "12:00-14:30,19:00-23:30",
            "saturday":  "12:00-15:00,19:00-23:30",
            "sunday":    "12:00-15:30" if random.random() < 0.6 else "CLOSED",
        },
        "tags": random.sample(
            ['terrasse', 'sans_gluten', 'vegan', 'halal', 'kosher',
             'livraison', 'réservation', 'wifi', 'parking_proche', 'vue'],
            k=random.randint(1, 4)
        ),
        "photos_count":  random.randint(0, 80),
        "last_updated":  (datetime.now() - timedelta(days=random.randint(0,90))).isoformat(),
    }

# Génération de 2000 restaurants via "API"
print("Simulation des appels API restaurants...")
api_responses = [simulate_api_response_restaurant(i) for i in range(1, 2001)]

# Flatten de la structure JSON (comme on le ferait sur une vraie réponse API)
restaurants_api = pd.json_normalize(api_responses, sep='_')
restaurants_api = restaurants_api.rename(columns={
    'address_quartier': 'quartier',
    'address_zip_code': 'code_postal',
    'location_lat': 'latitude',
    'location_lon': 'longitude',
})

print(f"Source A (API) : {restaurants_api.shape}")
print(restaurants_api[['id', 'name', 'cuisine', 'rating', 'price_range', 'quartier']].head(3).to_string())
```

```python
# ═══════════════════════════════════════════════════════════════════════════
# SOURCE B — Simulation d'un export Kaggle (style Zomato Dataset)
# Structure réelle : https://www.kaggle.com/datasets/shrutimehta/zomato-restaurants-data
# Le dataset Kaggle a ses propres colonnes et conventions
# ═══════════════════════════════════════════════════════════════════════════

def simulate_kaggle_row(resto_id):
    """
    Simule une ligne du Zomato-like Kaggle dataset.
    Ce dataset a une structure différente : noms de colonnes différents,
    notations sur 5 étoiles vs sur 10, etc.
    """
    quartier = random.choice(list(QUARTIERS_PARIS.keys()))
    
    # Note sur 5 (vs sur 5 dans l'API — mais attention aux décimales)
    note_kaggle = round(random.uniform(1.0, 5.0), 1)
    
    # Votes (équivalent des reviews)
    votes = max(0, int(np.random.lognormal(mean=np.log(80), sigma=1.0)))
    
    # Fourchette de prix en euros (pas en €/€€/€€€)
    prix_min = random.choice([8, 10, 12, 15, 18, 20, 25, 30])
    prix_max = prix_min + random.choice([5, 8, 10, 15, 20])
    
    return {
        'Restaurant ID':        f"ZOM_{resto_id:05d}",    # Convention Kaggle : espaces dans colonnes
        'Restaurant Name':      f"Le {fake.last_name()} {random.choice(['', 'Bistro', 'Café'])}",
        'Locality':             quartier,
        'Average Cost for two': random.randint(20, 120),  # En euros pour 2 personnes
        'Has Table booking':    random.choice(['Yes', 'No']),
        'Has Online delivery':  random.choice(['Yes', 'No']),
        'Is delivering now':    random.choice(['Yes', 'No']),
        'Aggregate rating':     note_kaggle,
        'Rating text':          random.choice(['Excellent', 'Very Good', 'Good', 'Average', 'Poor']),
        'Votes':                votes,
        'Cuisines':             ', '.join(random.sample(CUISINES, k=random.randint(1,2))),
        'Currency':             'Euro(s)',
        'City':                 'Paris',
        # Champ mal rempli dans le vrai Kaggle — simule les données manquantes
        'Address':              fake.street_address() if random.random() > 0.12 else '',
        # Parfois doublons dans les Kaggle (15% de lignes dupliquées simulées)
        '_is_duplicate':        random.random() < 0.08,
    }

kaggle_raw = pd.DataFrame([simulate_kaggle_row(i) for i in range(1, 1801)])

# Nettoyage typique d'un dataset Kaggle
def clean_kaggle_dataset(df):
    """
    Nettoyage standard d'un export Kaggle immature.
    Ce code documente les problèmes attendus.
    """
    df = df.copy()
    
    # Standardiser les noms de colonnes (espaces → underscores, minuscules)
    df.columns = (df.columns
                    .str.lower()
                    .str.replace(' ', '_', regex=False)
                    .str.replace('(', '', regex=False)
                    .str.replace(')', '', regex=False))
    
    # Supprimer les vrais doublons
    n_before = len(df)
    df = df[~df['_is_duplicate']].drop(columns=['_is_duplicate'])
    print(f"Doublons supprimés : {n_before - len(df)}")
    
    # Normaliser les champs Yes/No → booléens
    for col in ['has_table_booking', 'has_online_delivery', 'is_delivering_now']:
        df[col] = df[col].map({'Yes': True, 'No': False})
    
    # Gestion des adresses vides
    df['address'] = df['address'].replace('', np.nan)
    
    # Renommer pour harmonisation avec la source A
    df = df.rename(columns={
        'restaurant_id':     'kaggle_id',
        'restaurant_name':   'name_kaggle',
        'locality':          'quartier',
        'aggregate_rating':  'rating_kaggle',
        'cuisines':          'cuisines_text',
    })
    
    return df

kaggle_clean = clean_kaggle_dataset(kaggle_raw)
print(f"\nSource B (Kaggle) après nettoyage : {kaggle_clean.shape}")
print(f"Valeurs manquantes :")
print(kaggle_clean.isnull().sum()[kaggle_clean.isnull().sum() > 0])
```

```python
# ═══════════════════════════════════════════════════════════════════════════
# SOURCE C — Enrichissement : données météo simulées (style OpenWeather API)
# Enrichir les données avec la météo du jour de la visite
# ═══════════════════════════════════════════════════════════════════════════

def get_weather_data_simulated(date_str, lat=48.856, lon=2.352):
    """
    Simule une réponse de l'API OpenWeatherMap.
    En production : appel à https://api.openweathermap.org/data/2.5/history
    
    Paramètres :
        date_str : 'YYYY-MM-DD'
        lat, lon : coordonnées GPS
    
    Retourne : dict avec les clés de l'API réelle
    """
    date = datetime.strptime(date_str, '%Y-%m-%d')
    mois = date.month
    
    # Saisonnalité de Paris
    temp_saisonniere = {
        1: 5, 2: 6, 3: 10, 4: 13, 5: 17,
        6: 21, 7: 24, 8: 23, 9: 19, 10: 14,
        11: 9, 12: 6
    }
    temp_base = temp_saisonniere[mois]
    
    # Ajout de variabilité réaliste
    temp = round(temp_base + np.random.normal(0, 3), 1)
    
    # Probabilité de pluie par mois (Paris)
    pluie_prob = {1:0.30, 2:0.25, 3:0.28, 4:0.30, 5:0.28, 6:0.22,
                  7:0.18, 8:0.20, 9:0.28, 10:0.32, 11:0.33, 12:0.32}
    
    has_rain = random.random() < pluie_prob[mois]
    
    return {
        'dt':           int(date.timestamp()),
        'date':         date_str,
        'latitude':     lat,
        'longitude':    lon,
        'temp_celsius': temp,
        'feels_like':   round(temp - random.uniform(1, 4), 1),
        'humidity':     random.randint(45, 95),
        'weather_main': 'Rain' if has_rain else random.choice(['Clear', 'Clouds', 'Mist']),
        'wind_speed':   round(abs(np.random.normal(15, 8)), 1),
        'rain_mm':      round(random.uniform(0.5, 8.0), 1) if has_rain else 0.0,
        'uv_index':     round(random.uniform(0, 8) * (1 - has_rain * 0.6), 1),
    }

# Générer les données météo pour 90 jours
print("\nSimulation des appels API météo...")
dates_90j = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(90)]
weather_data = [get_weather_data_simulated(d) for d in dates_90j]
df_weather = pd.DataFrame(weather_data)

print(f"Source C (Météo API) : {df_weather.shape}")
print(f"Température moyenne : {df_weather['temp_celsius'].mean():.1f}°C")
print(f"Jours de pluie : {(df_weather['weather_main'] == 'Rain').sum()}/90")
```

---

### 3.2 Fusion des trois sources

```python
# ═══════════════════════════════════════════════════════════════════════════
# JOINTURE MULTI-SOURCES
# Le vrai défi : les sources n'ont pas les mêmes clés de jointure !
# ═══════════════════════════════════════════════════════════════════════════

# Simuler des visites de restaurants (logs utilisateurs)
def generer_visites(n=8000):
    """Simule les logs de visites utilisateurs — table de faits centrale."""
    visites = []
    
    for _ in range(n):
        # Date de visite dans les 90 derniers jours
        date_visite = (datetime.now() - timedelta(days=random.randint(0, 89))).strftime('%Y-%m-%d')
        
        # Restaurant depuis la source A (API)
        resto_id = f"rest_{random.randint(1, 2000):06d}"
        
        # Heure de visite
        heure = random.choices(
            [11, 12, 13, 14, 19, 20, 21, 22],
            weights=[0.05, 0.20, 0.25, 0.10, 0.08, 0.15, 0.12, 0.05]
        )[0]
        
        visites.append({
            'user_id':      f"USR_{random.randint(1, 500):04d}",
            'resto_id':     resto_id,
            'date_visite':  date_visite,
            'heure':        heure,
            'note_donnee':  random.choices([1,2,3,4,5], weights=[0.05,0.08,0.15,0.35,0.37])[0],
            'montant_euro': round(abs(np.random.normal(35, 15)), 2),
            'nb_personnes': random.choices([1,2,3,4,5,6], weights=[0.15,0.40,0.20,0.15,0.07,0.03])[0],
        })
    
    return pd.DataFrame(visites)

df_visites = generer_visites(8000)

# ─────────────────────────────────────────────────────────────────────────────
# JOINTURE COMPLÈTE
# ─────────────────────────────────────────────────────────────────────────────

# 1. Visites + Restaurants (Source A)
df_enrichi = df_visites.merge(
    restaurants_api[['id', 'name', 'cuisine', 'rating', 'price_range',
                      'quartier', 'latitude', 'longitude', 'review_count']],
    left_on='resto_id', right_on='id',
    how='left'
)

# 2. Jointure avec la météo du jour de visite
df_enrichi = df_enrichi.merge(
    df_weather[['date', 'temp_celsius', 'weather_main', 'rain_mm', 'humidity']],
    left_on='date_visite', right_on='date',
    how='left'
)

# 3. Feature engineering multi-sources
df_enrichi['is_weekend'] = pd.to_datetime(df_enrichi['date_visite']).dt.dayofweek >= 5
df_enrichi['is_raining']  = df_enrichi['weather_main'] == 'Rain'
df_enrichi['is_dinner']   = df_enrichi['heure'] >= 18
df_enrichi['depense_par_personne'] = df_enrichi['montant_euro'] / df_enrichi['nb_personnes']

# Variable cible : l'utilisateur a-t-il laissé une bonne note (≥4) ?
df_enrichi['satisfait'] = (df_enrichi['note_donnee'] >= 4).astype(int)

print("\nDataset final enrichi :")
print(f"Shape : {df_enrichi.shape}")
print(f"\nTaux de satisfaction global : {df_enrichi['satisfait'].mean():.1%}")

# Impact de la météo sur la satisfaction (analyse inter-sources)
print("\nImpact de la météo sur la satisfaction :")
print(df_enrichi.groupby('is_raining')['satisfait'].agg(['mean', 'count']).round(3))

print("\nSatisfaction par type de cuisine (top 5) :")
print(df_enrichi.groupby('cuisine')['satisfait'].mean()
      .sort_values(ascending=False).head(5).round(3))
```

---

### 3.3 Documentation de la traçabilité des sources

```python
# ─────────────────────────────────────────────────────────────────────────────
# DATA LINEAGE — Obligatoire pour les projets multi-sources
# Chaque colonne doit savoir d'où elle vient.
# ─────────────────────────────────────────────────────────────────────────────

DATA_LINEAGE = {
    # Colonne: (source, colonne_originale, transformation, hypothèse)
    'note_donnee':           ('Logs utilisateurs', 'rating_given',   'None',             'Notes 1-5 étoiles'),
    'montant_euro':          ('Logs utilisateurs', 'amount',          'None',             'Montant en euros TTC'),
    'cuisine':               ('API Restaurants',   'cuisine',         'None',             'Catégorie principale'),
    'rating':                ('API Restaurants',   'rating',          'None',             'Note agrégée 1-5'),
    'temp_celsius':          ('API Météo',          'main.temp',       '-273.15 (Kelvin)', 'Température à Paris 12h'),
    'is_raining':            ('API Météo',          'weather.main',    "== 'Rain'",        'Booléen pluie'),
    'depense_par_personne':  ('Calculée',           None,              'montant/personnes', 'Feature engineerée'),
    'satisfait':             ('Calculée',           None,              'note >= 4',         'Variable cible'),
}

print("\n" + "="*80)
print("DATA LINEAGE — Traçabilité des colonnes")
print("="*80)
print(f"{'Colonne':<28} {'Source':<20} {'Origine':<18} {'Note'}")
print("-"*80)
for col, (source, origine, transfo, note) in DATA_LINEAGE.items():
    print(f"  {col:<26} {source:<20} {origine or 'N/A':<18} {note}")
```

---

## 4. Validation croisée — est-ce que votre simulation est crédible ? {#validation}

```python
# ─────────────────────────────────────────────────────────────────────────────
# FRAMEWORK DE VALIDATION GÉNÉRALE
# Applicable à n'importe lequel des 3 cas
# ─────────────────────────────────────────────────────────────────────────────

def validation_report(df, nom_dataset, checks_metier):
    """
    Génère un rapport de validation structuré.
    
    Args:
        df              : DataFrame à valider
        nom_dataset     : Nom descriptif (pour le rapport)
        checks_metier   : liste de (description, résultat_bool)
    """
    print(f"\n{'='*65}")
    print(f"RAPPORT DE VALIDATION — {nom_dataset}")
    print(f"{'='*65}")
    print(f"  Lignes     : {len(df):,}")
    print(f"  Colonnes   : {df.shape[1]}")
    print(f"  Duplicates : {df.duplicated().sum()}")
    print(f"  Mémoire    : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    
    print(f"\n  Valeurs manquantes :")
    missing = df.isnull().mean()
    if missing.sum() == 0:
        print("    Aucune valeur manquante")
    else:
        for col, pct in missing[missing > 0].items():
            status = "⚠️" if pct > 0.10 else "✓"
            print(f"    {status} {col}: {pct:.1%}")
    
    print(f"\n  Checks métier :")
    n_pass = 0
    for desc, result in checks_metier:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"    {status} — {desc}")
        if result:
            n_pass += 1
    
    score = n_pass / len(checks_metier)
    verdict = "✅ CRÉDIBLE" if score >= 0.8 else "⚠️ À REVOIR" if score >= 0.6 else "❌ PROBLÉMATIQUE"
    print(f"\n  Score : {n_pass}/{len(checks_metier)} ({score:.0%}) → {verdict}")
    return score

# Exemple d'utilisation sur le dataset médical
checks_medical = [
    ("Taux no-show entre 10% et 30%", 
     0.10 <= df_medical_clean['no_show'].mean() <= 0.30),
    ("Psychiatrie a le taux le plus élevé",
     df_medical_clean.groupby('specialite')['no_show'].mean().idxmax() == 'Psychiatrie'),
    ("Corrélation délai ↔ no-show positive",
     df_medical_clean['delai_rdv_jours'].corr(df_medical_clean['no_show']) > 0),
    ("% confirmation envoyée entre 50% et 80%",
     0.50 <= df_medical_clean['confirmation_envoyee'].mean() <= 0.80),
    ("Âge moyen des patients entre 35 et 55 ans",
     35 <= df_medical_clean['age_patient'].median() <= 55),
    ("Pas de délais négatifs",
     (df_medical_clean['delai_rdv_jours'] >= 0).all()),
]

validation_report(df_medical_clean, "Dataset Médical (No-show)", checks_medical)
```

---

## 5. Checklist finale {#checklist}

```
AVANT DE COMMENCER LA SIMULATION
  ☐ Avez-vous documenté la structure exacte attendue des vraies données ?
  ☐ Avez-vous identifié au moins 3 sources de statistiques pour calibrer ?
  ☐ Les contraintes métier sont-elles codées (âges valides, prix positifs...) ?

PENDANT LA SIMULATION
  ☐ Chaque hypothèse est-elle commentée dans le code avec sa source ?
  ☐ Les corrélations inter-variables sont-elles encodées (pas juste indépendantes) ?
  ☐ Des imperfections réalistes sont-elles ajoutées (NaN, erreurs, doublons) ?
  ☐ La variable cible est-elle générée DEPUIS les features (et pas indépendamment) ?

VALIDATION DE LA SIMULATION
  ☐ Les distributions marginales correspondent-elles à des benchmarks ?
  ☐ Les corrélations connues sont-elles présentes dans le dataset généré ?
  ☐ Un expert métier a-t-il regardé le dataset et jugé les chiffres "crédibles" ?
  ☐ Le rapport de validation automatique passe-t-il à ≥80% ?

PASSAGE AUX VRAIES DONNÉES
  ☐ Le pipeline utilise-t-il un load_data() abstrait (swap facile) ?
  ☐ Les performances du modèle sur vraies données ont-elles été comparées ?
  ☐ Les hypothèses invalidées par les vraies données ont-elles été documentées ?
  ☐ Le rapport final mentionne-t-il clairement que les proto-résultats = données simulées ?

COMMUNICATION
  ☐ Chaque graphique, tableau, et métrique produit sur données simulées
    porte la mention visible "DONNÉES SIMULÉES — À REVALIDER"
  ☐ Les décisions business basées sur ces résultats sont-elles différées
    jusqu'à validation sur vraies données ?
```

---

## Conclusion

Travailler sans données réelles n'est pas un handicap — c'est une compétence.

Les trois cas couverts ici suivent tous le même principe fondamental :

> **Ne jamais inventer des données. Toujours modéliser des hypothèses explicites.**

La différence entre un prototype fragile et un projet solide, c'est que dans le second, n'importe qui peut lire votre code et savoir exactement pourquoi vous avez choisi `np.random.lognormal(mean=np.log(65), sigma=0.35)` plutôt que `np.random.normal(65, 20)` — et ce que ça implique pour la validité de vos conclusions.

---

*Pour aller plus loin :*
- **SDV (Synthetic Data Vault)** : génération synthétique à partir d'un vrai échantillon → `pip install sdv`
- **Faker** : génération de données textuelles réalistes → `pip install faker`
- **Great Expectations** : framework de validation de données en production
- **Mimesis** : alternative à Faker, plus rapide
- **CTGAN / TVAE** : génération avec GAN pour données tabulaires complexes
