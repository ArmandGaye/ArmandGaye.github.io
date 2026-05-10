# Acquisition de données

Ce README est la page de navigation du sous-répertoire.

## Notebooks du sous-répertoire

- [acquisition_donnees.ipynb](https://nbviewer.org/github/ArmandGaye/ArmandGaye.github.io/blob/main/data/notebooks/acquisition-donnees/acquisition_donnees.ipynb)

Désigne l’ensemble des méthodes et outils utilisés pour recueillir des informations brutes afin de les transformer ensuite en connaissances exploitables.

## 🔑 Les étapes principales

- **Définition des objectifs** : savoir pourquoi on collecte des données (ex. améliorer un produit, analyser un marché, suivre des performances).  
- **Choix des sources** : données internes (CRM, ventes, capteurs) ou externes (enquêtes, réseaux sociaux, open data).  
- **Méthodes de collecte** :  
  - Questionnaires et sondages  
  - Observations directes  
  - Capteurs et IoT  
  - Logs informatiques et bases de données 
  - Web scraping et API
- **Nettoyage et validation** : éliminer les doublons, corriger les erreurs, vérifier la cohérence.  
- **Stockage et organisation** : bases de données relationnelles, data lakes, ou systèmes cloud.

## ⚖️ Points essentiels à garder en tête

- **Qualité des données** : des données incomplètes ou biaisées mènent à de mauvaises décisions.  
- **Respect de la vie privée** : conformité aux réglementations (RGPD en Europe, par exemple).  
- **Sécurité** : protéger les données contre les accès non autorisés.

*En résumé, la collecte de données n’est pas seulement une opération technique : c’est une démarche stratégique qui conditionne la fiabilité des analyses et des décisions.*

Détaillons ensemble les cinq dimensions clés de l’acquisition de données :  
- Outils de collecte (logiciels)
- Architectures 
- Processus
- Bonnes pratiques
- Ethique

## 🛠️ Outils de collecte (logiciels)
- **ETL (Extract, Transform, Load)** : Talend, Pentaho, Informatica, Airbyte.  
- **Data scraping** : Scrapy, BeautifulSoup, Octoparse.  
- **IoT & capteurs** : Node-RED, Grafana, MQTT brokers.  
- **Questionnaires & enquêtes** : LimeSurvey, Qualtrics, Google Forms.  
- **Tracking web & produit** : Google Analytics, Matomo (open source), PostHog.  

---

## 🏗️ Architectures
- **Data Warehouse** : centralisation structurée (ex. Snowflake, BigQuery).  
- **Data Lake** : stockage brut et massif (Hadoop, Amazon S3).  
- **Lambda Architecture** : combinaison batch + streaming pour temps réel.  
- **Microservices & APIs** : ingestion via services modulaires.  
- **Edge Computing** : traitement proche des capteurs pour réduire la latence.  

---

## 🔄 Processus
1. **Définition des besoins** : objectifs clairs (marketing, recherche, monitoring).  
2. **Collecte** : via capteurs, formulaires, logs, scraping.  
3. **Prétraitement** : nettoyage, normalisation, suppression des doublons.  
4. **Stockage** : bases relationnelles (SQL) ou NoSQL (MongoDB, Cassandra).  
5. **Analyse & visualisation** : BI (Power BI, Tableau, Superset).  
6. **Diffusion** : partage des insights aux décideurs.  



## ⚖️ Éthique
- **Consentement** : informer les utilisateurs sur la collecte et obtenir leur accord.  
- **Confidentialité** : anonymiser les données sensibles.  
- **Transparence** : expliquer comment et pourquoi les données sont utilisées.  
- **Non-discrimination** : éviter les biais dans les datasets et les algorithmes.  
- **Conformité légale** : respecter RGPD en Europe, CDP au Sénégal, CCPA en Californie.  

*En résumé, l’acquisition de données est un équilibre entre **technologie (outils, architectures)** et **responsabilité (processus, bonnes pratiques, éthique)**.*


## Top 20 plateformes de téléchargement de données (open source et payantes).  

| Plateforme | Type | Licence / Prix | Lien officiel | Notes |  
| --- | --- | --- | --- | --- |  
| **Kaggle Datasets** | Gratuit | Open access | [kaggle.com/datasets](https://www.kaggle.com/datasets) | Communauté active, datasets variés |
| **Google Dataset Search** | Gratuit | Open access | datasetsearch.research.google.com [(datasetsearch.research.google.com in Bing)](https://www.bing.com/search?q="https%3A%2F%2Fdatasetsearch.research.google.com%2F") | Moteur de recherche global |
| **UCI ML Repository** | Gratuit | Open access | archive.ics.uci.edu/ml [(archive.ics.uci.edu in Bing)](https://www.bing.com/search?q="https%3A%2F%2Farchive.ics.uci.edu%2Fml") | Classique pour l’enseignement |
| **AWS Data Exchange** | Payant | Abonnement / à l’usage | aws.amazon.com/data-exchange [(aws.amazon.com in Bing)](https://www.bing.com/search?q="https%3A%2F%2Faws.amazon.com%2Fdata-exchange") | Données publiques américaine |
| **Statista** | Payant | Abonnement (≈59–199 €/mois) | [statista.com](https://www.statista.com) | Données publiques |
| **Bright Data** | Payant | À l’usage (datasets & scraping) | [brightdata.com](https://brightdata.com) | Web scraping & datasets |
| **Datarade** | Payant | Marketplace multi-fournisseurs | [datarade.ai](https://datarade.ai) | Données sectorielles |
| **Data.gov (USA)** | Gratuit | Open government data | [data.gov](https://data.gov) | Données publiques américaines |
| **EU Open Data Portal** | Gratuit | Open government data | [data.europa.eu](https://data.europa.eu) | Données européennes |
| **World Bank Data** | Gratuit | Open access | [data.worldbank.org](https://data.worldbank.org) | Indicateurs économiques |
| **UN Data** | Gratuit | Open access | [data.un.org](https://data.un.org) | Données globales |
| **Quandl (Nasdaq Data Link)** | Payant | Abonnement (≈49–299 €/mois) | [data.nasdaq.com](https://data.nasdaq.com) | Données financières |
| **Bloomberg Enterprise Data Catalog** | Payant | Premium (≥20k €/an) | [data.bloomberg.com](https://data.bloomberg.com) | Finance haut de gamme |
| **Zyte (ex Scrapinghub)** | Payant | À l’usage (API scraping) | [zyte.com](https://www.zyte.com) | E-commerce, web analytics |
| **Coresignal** | Payant | Abonnement (≈500–2000 €/mois) | [coresignal.com](https://coresignal.com) | Données RH & entreprises |
| **Data & Sons** | Payant | Marketplace | [dataandsons.com](https://dataandsons.com) | Données commerciales |
| **OpenStreetMap** | Gratuit | Open source (ODbL) | [openstreetmap.org](https://www.openstreetmap.org) | Données géospatiales |
| **Humanitarian Data Exchange (HDX)** | Gratuit | Open access | [data.humdata.org](https://data.humdata.org) | Données humanitaires |
| **GitHub (repos publics)** | Gratuit | Open access | [github.com](https://github.com) | Datasets partagés par les devs |
| **Zenodo** | Gratuit | Open access | [zenodo.org](https://zenodo.org) | Dépôt scientifique |

## 📌 Conseils pratiques
- **Gratuit/open source** : parfait pour l’apprentissage, la recherche académique et les projets exploratoires.  
- **Payant/premium** : recommandé pour les entreprises qui ont besoin de données fiables, actualisées et sectorielles (finance, marketing, RH).  
- **Attention aux licences** : certaines plateformes open data imposent des conditions d’utilisation (ODbL, CC-BY).  

## ⚠️ Limites et précautions
**Qualité variable** : les dépôts open source peuvent contenir des données incomplètes ou non mises à jour.

**Coût élevé**: les marketplaces premium facturent souvent par volume ou par abonnement.

**Conformité légale** : vérifier la conformité RGPD et les règles locales (au Sénégal, la CDP encadre la protection des données personnelles).