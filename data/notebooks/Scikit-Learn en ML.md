# [Qu’est-ce que Scikit-Learn en machine learning ?](https://www.nexa.fr/blog/quest-ce-que-scikit-learn-en-machine-learning)

Scikit-Learn est une bibliothèque Python dédiée au machine learning. Son interface simple et unifiée permet d’accéder à de nombreux algorithmes de classification, de régression, de clustering, de réduction de dimensionnalité et de sélection de modèles. Scikit-Learn est basée sur les bibliothèques NumPy, SciPy et matplotlib, qui sont des outils indispensables pour la manipulation de données et la visualisation en Python.
Qu’est-ce que Scikit-Learn ?

Scikit-Learn a été créée en 2007 par l’ingénieur David Cournapeau sous le nom de **scikits. Learn**, pour un projet Google Summer of Code (GSoC). Développé et maintenu par une communauté de bénévoles, le projet compte aujourd’hui plus de 30 contributeurs actifs. Son lancement public a eu lieu le premier février 2010, ce qui lui a permis de bénéficier du soutien financier de l’INRIA, Google, Tinyclues et Python Software Foundation.

**Scikit-Learn** est une bibliothèque open source d’apprentissage automatique en Python. Aussi connue sous son nom abrégé sklearn, elle fournit une vaste gamme d’outils, d’algorithmes et de fonctionnalités permettant de faciliter le développement de modèles de machine learning. Scikit-learn couvre les principaux algorithmes de machine learning généralistes : 
- classification, 
- régression, 
- clustering, 
- gradient boosting.

## À quoi sert la bibliothèque open source Scikit-Learn ?

Scikit-Learn est une bibliothèque open source sous licence BSD (Berkeley Software Distribution License) réputée pour sa documentation complète et sa communauté active qui fournit un soutien et des ressources supplémentaires. Scikit-Learn permet de construire des modèles prédictifs en s’appuyant sur des algorithmes de régressions linéaire et logistique, des Support Vector Machine (SVM), des arbres de décision, des forêts aléatoires. Les modèles prédictifs sont utiles dans les domaines du diagnostic médical, la détection de fraudes, ou encore l’autonomie des véhicules par exemple.

Cette bibliothèque puissante propose des métriques d’évaluation pour estimer la performance des modèles de machine learning. Elle prend en charge des techniques de validation croisée pour estimer la capacité de généralisation des modèles. Les développeurs y trouvent des outils de prétraitement des données pour nettoyer, normaliser et coder les variables catégorielles. La bibliothèque offre également la possibilité de créer des pipelines de traitement des données groupées.

## Le fonctionnement de la bibliothèque Scikit-Learn

Scikit-Learn est disponible notamment sur GitHub. Il suffit d’accéder au fichier du framework ML de Python et d’utiliser un gestionnaire de package sous Python (pip ou Conda). L’installation est possible depuis un environnement Windows, Linux ou macOS. Il est toutefois recommandé de l’installer depuis un gestionnaire de package conçu sous Python, pour optimiser la compatibilité entre système et framework :
- pip, qui permet de gérer les packages Python via PyPI (Python Package Index) avec la ligne de code suivante : « pip install -U scikit-learn » ;
- conda avec l’entrée « conda install -c anaconda scikit-learn ».

Scikit-Learn regroupe de nombreux algorithmes indispensables à une machine learning généraliste, comme les modes classification, clustering, gradient boosting, régression. La bibliothèque est également pourvue de NumPy, Matplotlib et SciPy, des bibliothèques spécialisées dans les calculs scientifiques. Scikit-Learn possède aussi des estimateurs, des objets Python représentant les algorithmes d’apprentissage automatique. Chacun expose une interface cohérente avec des méthodes standardisées telles que **«fit()»**, **«predict()»** et **«score()»**.

## Quelles sont les bases de la bibliothèque Scikit-Learn ?

L’API étant uniformisée, les quatre étapes de modélisation sont communes à l’utilisation d’un grand nombre de modèles disponibles dans la librairie. Dès lors que la logique de construction d’un modèle est comprise, elle peut être très facilement appliquée à d’autres modèles.

Commencez par choisir un modèle en important la classe appropriée de Scikit-Learn, puis paramétrez le modèle. Vous pouvez également renseigner vos paramètres manuellement. La librairie dispose de techniques comme GridSeachCV pour trouver les meilleurs paramètres. Vous allez pouvoir entraîner le modèle sur le jeu d’apprentissage en utilisant la méthode fit.

Le modèle est alors prêt à être testé sur de nouvelles données :

- En apprentissage **supervisé**, avec la méthode predict sur les données test.
- En apprentissage **non supervisé**, avec les méthodes *transform* ou *predict*.


## Quels sont les deux grands types d’apprentissage de Scikit-Learn ?

Deux types d’apprentissage jouent un rôle essentiel dans l’analyse et l’exploitation des données.<br> L’**apprentissage supervisé** *est utilisé lorsque l’on dispose de données étiquetées et que l’on souhaite effectuer des prédictions ou des classifications*.<br> L’**apprentissage non supervisé** *permet de découvrir des informations cachées ou des structures dans les données non étiquetées*. Scikit-Learn offre une large gamme d’outils et de modèles pour soutenir ces deux types d’apprentissage.

## L’apprentissage supervisé avec la bibliothèque Scikit-Learn

L’apprentissage supervisé est l’utilisation d’ensembles de données étiquetées pour former des algorithmes permettant de classer des données ou de prédire des résultats précis. Au fur et à mesure que les données d’entrée sont introduites dans le modèle, il ajuste ses poids grâce à un processus d’apprentissage par renforcement. L’apprentissage supervisé aide les organisations à résoudre des problèmes telles la classification et la régression.

- La **classification** : Les algorithmes utilisent les données d’entraînement en entrée pour prédire la probabilité que les suivantes entrent dans l’une des catégories définies. L’un des exemples les plus courants de classification est le filtrage des e-mails en « spam » ou « non-spam ».

- La **régression** : La régression linéaire est un algorithme utilisé pour prédire des valeurs dans une plage continue.

## L’apprentissage non supervisé avec la bibliothèque Scikit-Learn

L’apprentissage non supervisé est beaucoup plus complexe. Il utilise des algorithmes d’apprentissage automatique pour analyser et regrouper des ensembles de données non étiquetées. Son objectif est de découvrir des similarités dans les données ingérées sans intervention humaine. <br>Scikit-Learn propose plusieurs algorithmes d’apprentissage non supervisé :
- Le clustering (par exemple K-means et DBSCAN) permet de regrouper les données similaires. 
- La réduction de dimension (par exemple PCA) réduit la complexité des données tout en conservant leur information essentielle.

## Pourquoi utiliser la bibliothèque Scikit-Learn ?

Scikit-Learn offre une vaste gamme de modèles d’apprentissage automatique préimplémentés et dispose d’une collection complète d’algorithmes. De plus, elle a été conçue avec une interface simple et cohérente, la rendant très conviviale pour les utilisateurs y compris les novices. Ses API sont claires et bien documentées, ce qui permet de définir, d’entraîner et d’évaluer des modèles de manière intuitive.

Scikit-Learn propose de nombreuses fonctionnalités de prétraitement des données, comme la normalisation, l’encodage des variables catégorielles, la réduction de dimension et la gestion des valeurs manquantes. Cela représente un gain de temps significatif pour les développeurs. De plus, des outils intégrés permettent d’évaluer les performances des modèles. Elle propose des fonctionnalités pour la sélection de modèles. Sa recherche de grilles (Grid Search) permet d’explorer facilement différentes combinaisons d’hyperparamètres pour trouver les meilleures performances.

Scikit-Learn s’intègre parfaitement avec d’autres bibliothèques Python populaires comme NumPy et pandas, ce qui facilite l’importation, la manipulation et la transformation des données. Enfin, la bibliothèque Scikit-Learn bénéficie d’une documentation complète régulièrement mise à jour, comprenant des exemples, des tutoriels et des guides d’utilisation. C’est une bibliothèque fiable qui simplifie le processus de développement de modèles de machine learning. Elle permet d’obtenir des résultats précis et reproductibles.
## Qui utilise la bibliothèque d’apprentissage Scikit-Learn ?

Selon la fondation INRIA, Scikit-learn est utilisé régulièrement par plus d’un demi-million de personnes dans le monde ! Ainsi, elle est implémentée par des références dans le domaine de la data science, comme Dataiku, DataRobot ou encore Knime. Elle est aussi prise en charge par de plus en plus d’acteurs du cloud. On citera par exemple Google via son service Cloud Machine Learning Engine, IBM avec Watson Machine Learning ou encore Microsoft et d’Azure Machine Learning.

Sckit-learn est utilisé dans de nombreux secteurs comme outil principal pour la mise en place de modèles prédictifs. Des entreprises réputées l’utilisent comme outil principal dans leurs systèmes de recommandations et de prédictions des risques. Par exemple, l’INRIA l’utilise pour soutenir la recherche de pointe dans de nombreux domaines, comme la neuro-imagerie, la vision par ordinateur, l’analyse d’images médicales ou encore la sécurité.

Scikit-learn est beaucoup utilisé pour les recommandations musicales de Spotify, mais également pour la recommandation d’hôtels et de destination de Booking.com. Même Netflix l’utilise (ainsi que numpy, scipy, matplotlib, pandas ou cvxpy) pour ses algorithmes de recommandation, d’illustrations personnalisées et de marketing. La communauté interne de développeurs et de data scientists de BNP Paribas Cardif utilise des pipelines de scikit-learn dans leur gouvernance interne du risque de modèle pour diminuer les risques opérationnels et le risque d’overfitting.
Les avantages et les inconvénients de Scikit-Learn

Le premier avantage de Scikit-Learn est sa simplicité d’utilisation. Le package est extrêmement adaptable et utile, et peut servir des objectifs réels comme la prédiction du comportement des consommateurs ou encore le développement de neuroimages. Une documentation APU détaillée permet aux utilisateurs de connecter facilement les algorithmes à leurs plateformes. Enfin, Scikit-learn est soutenu et maintenu à jour par de nombreux contributeurs et une grande communauté mondiale.

En revanche, si Scikit-Learn est efficace pour bâtir des modèles prédictifs, il n’est pas idéal pour la simple lecture, la manipulation et la synthèse d’information. Sur ces points, nous lui préfèrerons des librairies telles que NumPy ou Pandas, avec lesquelles Scikit-Learn fonctionne d’ailleurs parfaitement.
Comment apprendre à utiliser Scikit-Learn ?

Pour utiliser Scikit-Learn, des bases en python et en traitement de données sont évidemment indispensables. Vous pourrez ensuite vous former à l’utilisation sur le site officiel de Scikit-Learn, qui propose une documentation complète, des tutoriels, des exemples et une FAQ. Toutes ces ressources vous permettront de mettre en pratique et d’expérimenter avec différents modèles et fonctionnalités de Scikit-Learn.

En plus de la documentation officielle, de nombreuses ressources en ligne permettent d’apprendre et de partager des connaissances sur Scikit-Learn. Blogs, tutoriels vidéo et forums de discussion sont autant de ressources pour approfondir votre compréhension et découvrir de nouvelles astuces et techniques. Vous pouvez également trouver sur le site de l’INRA un MOOC totalement gratuit.

Scikit-Learn est donc un outil incontournable pour tout développeur ou chercheur qui souhaite découvrir ou approfondir ses connaissances en machine learning. Que ce soit pour réaliser des analyses exploratoires, préparer des données, entraîner et évaluer des modèles, ou encore déployer des solutions, Scikit-Learn offre une solution adaptée à chaque étape du processus.