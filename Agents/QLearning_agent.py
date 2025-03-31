import numpy as np
from itertools import product
from event_from_obs import event_from_obs_gym
import pickle as pkl

def compute_epsilon_decay(eps_start, eps_end, num_episodes):

    epsilon_decay = (eps_end / eps_start) ** (1 / num_episodes)
    return epsilon_decay

class QLearningAgent():
    def __init__(self, env, alpha=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, all_types=None, all_attributes=None):
        """
        Args:
            env: l'environnement OpenAI Gym-like.
            alpha: taux d'apprentissage
            gamma: facteur de discount
            epsilon: probabilité initiale de choisir une action aléatoire (epsilon-greedy)
            epsilon_min: valeur minimale d'epsilon
            epsilon_decay: facteur de décroissance d'epsilon
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min

        # Liste de tous les symboles manipulés (états possibles).
        # 'all_types' doit être défini ou importé selon votre code existant.
        # Exemple : all_types = ["door_opened", "chest_opened", "key_found", ...]
        self.all_types = all_types
        self.all_attributes = all_attributes

        # Génère toutes les actions possibles (3 bits => 8 actions possibles).
        # Chaque action est un tuple, par exemple (0, 1, 0).
        self.action_set = list(product([0, 1], repeat=3))  # [(0,0,0), (0,0,1), ..., (1,1,1)]
        self.num_actions = len(self.action_set)  # 8

        # On construit une table Q : pour chaque symbole (état), un tableau de 8 valeurs Q.
        self.q_table = {sym: np.zeros(self.num_actions) for sym in self.all_types}

    def _epsilon_greedy_action(self, symbol):
        """
        Sélectionne l'action selon une politique epsilon-greedy :
        - Avec probabilité epsilon, on choisit une action aléatoire
        - Sinon, on prend l'action avec la Q-value maximale
        """
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(0, self.num_actions)
        else:
            action_idx = np.argmax(self.q_table[symbol])
        return action_idx

    def _get_symbol_from_obs(self, obs):
        """
        Extrait le symbole de l'événement depuis l'observation Gym.
        Vous pouvez aussi directement utiliser `event_from_obs_gym`
        si cela vous donne déjà `{'symbol': ...}`.
        """
        event = event_from_obs_gym(obs, self.all_types, self.all_attributes)
        return event["symbol"]

    def train(self, num_episodes=10):
        """
        Entraîne l’agent avec Q-learning, en appliquant la formule de mise à jour de la Q-table.
        """
        self.epsilon_decay = compute_epsilon_decay(self.epsilon, self.epsilon_min, num_episodes)
        for episode in range(num_episodes):
            obs = self.env.reset()
            done = False

            # État courant (symbol)
            current_symbol = self._get_symbol_from_obs(obs)
            episode_reward = 0

            while not done:
                # Sélection d'action (index d'action) via epsilon-greedy
                action_idx = self._epsilon_greedy_action(current_symbol)
                action = self.action_set[action_idx]  # On récupère le tuple binaire (ex: (1, 0, 1))

                # Exécute l'action dans l'environnement
                next_obs, reward, done, info = self.env.step(action)
                episode_reward += reward

                # Récupère le symbole suivant
                next_symbol = self._get_symbol_from_obs(next_obs)

                # Mise à jour Q-learning
                current_q = self.q_table[current_symbol][action_idx]

                # Si terminé, la cible est simplement 'reward'
                if done:
                    target = reward
                else:
                    # On prend la Q-value max de l'état suivant
                    max_next_q = np.max(self.q_table[next_symbol])
                    target = reward + self.gamma * max_next_q

                # Nouvelle valeur Q(s, a)
                new_q = current_q + self.alpha * (target - current_q)
                self.q_table[current_symbol][action_idx] = new_q

                # On passe à l'état suivant
                current_symbol = next_symbol

            # Mettre à jour epsilon (décroissance)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            if episode%10==0:
              print(f"Episode {episode+1}/{num_episodes}, Reward = {episode_reward}, Epsilon = {self.epsilon:.3f}")

    def test(self, num_episodes=1):
        """
        Teste la politique (greedy) après entraînement.
        """
        for ep in range(num_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0

            current_symbol = self._get_symbol_from_obs(obs)

            print(f"\n----- TEST EPISODE {ep+1} -----")

            while not done:
                # Choix de l'action de façon purement greedye (Q-max)
                action_idx = np.argmax(self.q_table[current_symbol])
                action = self.action_set[action_idx]

                next_obs, reward, done, info = self.env.step(action)
                episode_reward += reward

                print(f"Event: {current_symbol}, Action: {action}, Reward: {reward}")

                current_symbol = self._get_symbol_from_obs(next_obs)

            print(f"Episode terminé avec un total de reward : {episode_reward}")

    def print_q_table(self):
        """
        Affiche la Q-table pour chaque symbole et ses valeurs Q.
        """
        for symbol, q_values in self.q_table.items():
            print(f"Symbol: {symbol}, Q-values: {q_values}")
            
    def save(self, filename):
        """
        Enregistre l'agent
        """
        with open(filename, 'wb') as f:
            pkl.dump(self, f)
            f.close()
            
    def load(self, filename):
        """
        Charge l'agent
        """
        with open(filename, 'rb') as f:
            agent = pkl.load(f)
            f.close()
        
        return agent
