import gym
from collections import deque
import numpy as np
from event_from_obs import event_from_obs_gym

class TimeWindowWrapper(gym.Wrapper):
    def __init__(self, env, window_size=4, types=None, attributes=None):
        super().__init__(env)
        self.window_size = window_size
        self.types = types
        self.attributes = attributes
        self.symbol_dim = len(types)  # one-hot encoding
        self.event_dim = self.symbol_dim + 2  # one-hot(symbol) + start + end

        # Buffer of processed events
        self.events_buffer = deque(maxlen=window_size)

        # Define observation space
        # Total dim = window_size * event_dim
        obs_low = np.zeros((self.window_size, self.event_dim), dtype=np.float32)
        obs_high = np.ones((self.window_size, self.symbol_dim), dtype=np.float32)
        time_high = np.array([[500, 500]] * self.window_size, dtype=np.float32)  # assuming time is normalized
        obs_high = np.concatenate([obs_high, time_high], axis=1)

        self.observation_space = gym.spaces.Box(
            low=obs_low.flatten(),
            high=obs_high.flatten(),
            dtype=np.float32
        )

    def reset(self):
        obs = self.env.reset()
        self.events_buffer.clear()

        evt_info = event_from_obs_gym(obs, self.types, self.attributes)
        self._store_event(evt_info)

        return self._get_observation()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        evt_info = event_from_obs_gym(obs, self.types, self.attributes)
        self._store_event(evt_info)

        return self._get_observation(), reward, done, info

    def _store_event(self, event_info):
        """Convert event info into a numeric vector and store it."""
        symbol_idx = self.types.index(event_info["symbol"])
        one_hot_symbol = np.zeros(self.symbol_dim, dtype=np.float32)
        one_hot_symbol[symbol_idx] = 1.0

        start_t = float(event_info["start_time"]) # Normalize timestamps
        end_t   = float(event_info["end_time"])

        numeric_evt = np.concatenate([one_hot_symbol, [start_t, end_t]])
        self.events_buffer.append(numeric_evt)

    def _get_observation(self):
        """
        Returns a flattened vector of length (window_size * event_dim).
        Pads with zeros if fewer than window_size events have occurred.
        """
        out = []

        for i in range(self.window_size):
            if i < len(self.events_buffer):
                out.append(self.events_buffer[i])
            else:
                out.append(np.zeros(self.event_dim, dtype=np.float32))

        return np.concatenate(out, axis=0)
