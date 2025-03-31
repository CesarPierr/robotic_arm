import gym
import numpy as np
from event_from_obs import event_from_obs_gym
# For vectorization
from stable_baselines3.common.vec_env import DummyVecEnv
# Recurrent PPO from SB3-contrib
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy

# ---------------------------------------------------------------------
# Your custom wrapper (unchanged), but make sure you have np imported
# ---------------------------------------------------------------------
class LSTMCompatibleWrapper(gym.Wrapper):
    """
    - No longer maintains a rolling 4-event buffer.
    - Returns [symbol_idx, start_time, end_time].
    """
    def __init__(self, env, types, attributes):
        super().__init__(env)
        self.types = types
        self.attributes = attributes
        self.obs_dim = len(types) + 2  # symbol_idx + start_time + end_time
        self.max_time = 100000
        high = np.array([1.0] * len(types) + [1.0, 1.0], dtype=np.float32)
        low  = np.array([0.0] * len(types) + [0.0, 0.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self):
        obs = self.env.reset()
        evt_info = event_from_obs_gym(obs, self.types, self.attributes)
        return self._get_observation(evt_info)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        evt_info = event_from_obs_gym(obs, self.types, self.attributes)
        return self._get_observation(evt_info), reward, done, info

    def _get_observation(self, evt_info):
        symbol_idx = self.types.index(evt_info["symbol"])
        one_hot_symbol = np.zeros(len(self.types), dtype=np.float32)
        one_hot_symbol[symbol_idx] = 1.0

        start_t = float(evt_info["start_time"])/self.max_time  
        end_t = float(evt_info["end_time"])/self.max_time 

        return np.concatenate([one_hot_symbol, [start_t, end_t]]).astype(np.float32)



# ---------------------------------------------------------------------
# Example "factory" function for creating one instance of the wrapped env
# ---------------------------------------------------------------------
def make_wrapped_env(types, attributes):
    """
    Returns a function that, when called, creates a new instance
    of your custom environment wrapped with LSTMCompatibleWrapper.
    """
    def _init():
        env = gym.make("OpenTheChests-v2")
        env = LSTMCompatibleWrapper(env, types=types, attributes=attributes)
        return env
    return _init

from stable_baselines3.common.callbacks import BaseCallback

class PrintEpisodeRewardCallback(BaseCallback):
    """
    Custom callback for printing episode rewards every `print_freq` episodes.
    Works when using multiple parallel environments.
    """
    def __init__(self, print_freq=10, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq
        # Will be initialized later (when callback is first called),
        # because `training_env` isn't available until the callback is bound
        self.episode_rewards = None
        self.episodes_count = 0

    def _init_callback(self) -> None:
        """
        Called at the start of training to initialize needed values.
        """
        # One reward tracker per parallel environment
        self.episode_rewards = np.zeros(self.training_env.num_envs)

    def _on_step(self) -> bool:
        """
        Called at every environment step.
        """
        # Current rewards and done signals for each parallel env
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]

        # Accumulate episode rewards
        self.episode_rewards += rewards

        # For each environment that is done, we print the reward and reset
        for i, done in enumerate(dones):
            if done:
                self.episodes_count += 1
                if self.episodes_count % self.print_freq == 0:
                    print("#############################################")
                    print(f"Episode {self.episodes_count} reward: {self.episode_rewards[i]}")
                    print("#############################################")

                self.episode_rewards[i] = 0.0

        return True