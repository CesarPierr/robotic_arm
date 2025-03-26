
# Projet Robot KUKA pour la Manipulation de Boîtes

## Description
Ce projet propose un environnement de simulation développé avec Gymnasium, PyBullet et Stable Baselines3 pour entraîner un robot KUKA à atteindre une boîte cible parmi plusieurs. L'agent est récompensé en fonction de la proximité de son effecteur par rapport à la boîte choisie et pénalisé si la boîte se déplace accidentellement.

## Fonctionnalités
- Simulation 3D d'un robot KUKA dans un environnement avec plusieurs boîtes.
- Identification et mise en évidence de la boîte cible.
- Entraînement par apprentissage par renforcement utilisant l'algorithme PPO.
- Visualisation de l'exécution via un rendu image (GIF animé).

## Setup
1. **Création de l'environnement Conda**  
   Le fichier `environment.yaml` est fourni pour installer toutes les dépendances.  
   Exécutez les commandes suivantes dans un terminal :
   ```bash
   conda env create -f environment.yaml
   conda activate nom_de_votre_env
   ```
   (Remplacez `nom_de_votre_env` par le nom défini dans le fichier YAML.)

2. **Installation manuelle des dépendances (optionnel)**  
   Assurez-vous d'avoir installé :
   - Python 3.7+
   - gymnasium
   - pybullet
   - stable-baselines3
   - imageio

## How to Use
1. **Entraînement du modèle**  
   Lancez l'entraînement du robot en exécutant le script principal :
   ```bash
   python main.py
   ```
   Ce script crée plusieurs environnements parallèles et entraîne un modèle PPO pour faire apprendre au robot comment atteindre la boîte cible.

2. **Évaluation et Visualisation**  
   Un notebook Jupyter est fourni pour tester le modèle entraîné. Il permet de :
   - Charger un modèle sauvegardé.
   - Créer un environnement avec des boîtes.
   - Choisir une boîte cible via un paramètre.
   - Exécuter la politique du robot et générer un GIF animé montrant l'évolution.
   
   Pour utiliser le notebook, lancez :
   ```bash
   jupyter notebook
   ```
   Puis ouvrez le notebook correspondant et exécutez les cellules pour visualiser le comportement du robot.

## Structure du Projet
- `main.py` : Script principal pour l'entraînement du modèle.
- `environment.yaml` : Fichier de configuration pour créer l'environnement Conda.
- `notebook.ipynb` : Notebook Jupyter pour l'évaluation et la visualisation du modèle.
- `README.md` : Ce fichier.
