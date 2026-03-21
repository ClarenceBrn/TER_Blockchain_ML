# Bitcoin Deanonymization: A Machine Learning Cascading Approach


Projet de recherche étudiant (TER).


Ce dépôt contient l'implémentation d'un pipeline d'apprentissage automatique visant à désanonymiser et classifier les entités du réseau Bitcoin (Exchanges, Mixers, Gambling, etc.). Ce travail est basé sur la reproduction et l'adaptation de l'article de recherche : *Cascading Machine Learning to Attack Bitcoin Anonymity* (Zola et al., 2019).


## Architecture du projet


* **`features_engineering.ipynb`** : Script d'extraction Big Data. Parse les transactions brutes de la blockchain par chunks pour extraire les heuristiques, construire le graphe d'entités et générer les 1-motifs et 2-motifs.

* **`Cascading.ipynb`** : Le cœur du projet. Pipeline Machine Learning comparant un modèle de base (Baseline) avec l'approche par Cascade (enrichissement itératif des features locales vers les caractéristiques globales).

* **`utils.py`** : Fonctions utilitaires pour l'initialisation des modèles, la validation croisée stratifiée et la génération des rapports de métriques (Accuracy, MCC).

* **`data_gen.py`** : Générateur de données synthétiques (Mock Data). Modélise mathématiquement les distributions en loi de puissance du réseau Bitcoin pour tester le pipeline ML sans nécessiter les téraoctets de données brutes.


## Comment tester ce projet


Les données brutes de la blockchain étant trop volumineuses pour GitHub (fichiers de >150 Mo/jour), deux solutions s'offrent à vous pour évaluer le code :


1. **Aperçu des données réelles** : Un échantillon des structures de données (50 entités, 1000 transactions) est disponible dans le dossier `data_sample/`.

2. **Exécuter le pipeline ML** : Vous pouvez lancer `Cascading.ipynb` de bout en bout en utilisant la fonction de génération de données de `data_gen.py` incluse dans le notebook.


### Installation


```bash

git clone [https://github.com/ClarenceBrn/TER_Blockchain_ML.git](https://github.com/ClarenceBrn/TER_Blockchain_ML.git)

cd TER_Blockchain_ML

python -m venv .venv

source .venv/bin/activate