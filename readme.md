
# Projet Robot KUKA pour la Manipulation de Boîtes

## Description
Ce projet propose un environnement de simulation développé avec Gym, PyBullet et Stable Baselines3 pour entraîner un robot KUKA à atteindre une boîte cible parmi plusieurs. L'agent est récompensé en fonction de la proximité de son effecteur par rapport à la boîte choisie et pénalisé si la boîte se déplace accidentellement.

## Fonctionnalités
- Simulation 3D d'un robot KUKA dans un environnement avec plusieurs boîtes.
- Identification et mise en évidence de la boîte cible.
- Entraînement par apprentissage par renforcement utilisant l'algorithme PPO.
- Visualisation de l'exécution via un rendu image (GIF animé).
- Intégration avec l'environnement "Open the Chests" pour des scénarios de manipulation d'objets plus complexes.

## Intégration avec Open the Chests
Ce projet étend ses fonctionnalités en intégrant l'environnement "Open the Chests" (OtC), qui fournit un cadre de prise de décision séquentielle basé sur des événements.

### Caractéristiques de l'environnement OtC
- Environnement séquentiel où les décisions précédentes affectent les états futurs
- Système d'événements basé sur des règles temporelles
- Possibilité de définir des comportements complexes pour le robot

### Avantages de l'intégration
- Permet au robot de prendre des décisions basées sur des séquences d'événements
- Facilite l'apprentissage de comportements plus sophistiqués
- Offre un cadre pour tester des scénarios de manipulation d'objets plus complexes

### Flux de travail
1. Le système OtC génère des séquences d'événements
2. Ces événements sont transformés en commandes pour le robot KUKA
3. Le robot utilise son modèle entraîné pour interagir avec la boîte correspondante
4. Les performances sont évaluées en fonction de la capacité du robot à suivre correctement les instructions

## Setup
1. **Création de l'environnement**  
   Utilisez le fichier `requirements.txt` pour installer les dépendances dans un environnement virtuel Python :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Installation manuelle des dépendances (optionnel)**  
   Assurez-vous d'avoir installé :
   - Python 3.9+
   - gym
   - pybullet
   - stable-baselines3
   - imageio

## Structure du Projet
- `requirements.txt` : Fichier de configuration pour créer l'environnement Conda.
- `notebook.ipynb` : Notebook Jupyter pour l'évaluation et la visualisation du modèle.
- `README.md` : Ce fichier.
- `roboto.py`: The physical env wrapper
- `Agents\QLearning_agent.py` : The Q-Learning agent used for difficulty 0
- `Env_Wrappers\TimeWindowWrapper.py` : The environnement wrapper used for difficulty 1
- `Env_Wrappers\LSTMCompatibleWrapper.py` : The environnement wrapper used for difficulty 2
