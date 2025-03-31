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

        # We will store up to 4 “processed” events
        self.events_buffer = deque(maxlen=window_size)

        # ---- Construct a new observation space example. ----
        # The environment’s raw observation is presumably a dict with indices,
        # but we want to store four events each with (symbol, bg, fg, start, end).
        # For a minimal approach, you can flatten each event into a numeric vector
        # (e.g. symbol_idx, bg_idx, fg_idx, start, end), and then you have 5
        # numbers * 4 events = 20.
        # Alternatively, you can keep them as a more structured space
        # if you are using a custom policy. Here we'll create a simple Box space.

        # We'll define: event vector = [ e_type, bg, fg, start_time, end_time ]
        # Each is numeric, though the type might be discrete. For demonstration,
        # we assume they fit in [0, 100], but please refine as appropriate.
        high = np.array([len(types)-1, 1e6, 1e6]*window_size, dtype=np.float32)
        low  = np.array([0,  0,    0]*window_size, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self):
        obs = self.env.reset()
        self.events_buffer.clear()

        # We might get an initial “obs” that we want to store as an event.
        # Or, if your environment typically returns an event after the first step,
        # you can skip storing one here.
        evt_info = event_from_obs_gym(obs, self.types, self.attributes)
        self._store_event(evt_info)

        return self._get_observation()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Convert raw obs -> event
        evt_info = event_from_obs_gym(obs, self.types, self.attributes)
        self._store_event(evt_info)

        return self._get_observation(), reward, done, info

    def _store_event(self, event_info):
        """Maps the dictionary (symbol, start, end) to a numeric vector, then stores it."""
        # Convert symbol and color strings to integer indexes, or do a bigger dictionary lookup.
        # For simplicity, assume each mapping is a small dictionary:
        symbol_idx = self.types.index(event_info["symbol"])  # e.g. “A” -> 0, “B” -> 1, ...

        # start_time, end_time can remain floats; you might want to scale them.
        start_t = float(event_info["start_time"])
        end_t   = float(event_info["end_time"])

        numeric_evt = [symbol_idx, start_t, end_t]
        self.events_buffer.append(numeric_evt)

    def _get_observation(self):
        """
        Flatten all events in events_buffer into one 1D array of size 4*5 = 20 (if 4 events).
        If we have fewer than 4 stored so far (e.g. early in an episode), we’ll just zero-pad them.
        """
        # events_buffer is a deque of length <= 4
        # each element is a 5-dim list [ symbol_idx, bg_idx, fg_idx, start_t, end_t ]
        # We’ll produce a length-4 list of these 5-dim vectors,
        # or zero if the event is not present.

        # Build a list of 4 vectors, from oldest to newest
        out = []
        for i in range(self.window_size):
            if i < len(self.events_buffer):
                out.append(self.events_buffer[i])
            else:
                out.append([0, 0.0, 0.0])  # zero fill if we have fewer than 4 so far

        # Flatten to shape (20,)
        return np.array(out, dtype=np.float32).reshape(-1)