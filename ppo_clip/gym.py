from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from utils import MAX_EXTRA_VARS, ACTION_CATEGORIES, ACTION_TYPES
from gymnasium.spaces import MultiDiscrete
from vectorize_game_state import game_state_to_vector

class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3


class TribesEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    _game_state = None

    def __init__(self, render_mode=None, size=11):
        self.size = size  # The size of the square grid

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "spatial_tensor": spaces.Box(0, size**2 - 1, shape=(size, size, 27), dtype=int),
                "global_info": spaces.Box(0, size**2 - 1, shape=(2,), dtype=int),
            }
        )

        self.action_space_shape = (len(ACTION_CATEGORIES), max(ACTION_TYPES.values()) + 1, size, size, size, size, MAX_EXTRA_VARS)
        self.action_space = MultiDiscrete(self.action_space_shape)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def get_game_state(self):
        # Return network response to Java
        # fetch..
        def fetch_tribes():
            pass

        self._game_state = fetch_tribes()
        self._vector_game_state, self._info = game_state_to_vector(self._game_state)


    def _get_obs(self):
        return {"spatial_tensor": self._game_state, "global_info": self._info}

    def _get_info(self):
        return {
            "global_info": self._info
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Create a new game
        self.get_game_state()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # Send action to Java
        self.send_action(action)

        terminated = game_over(self._game_state)
        reward = reward(self._game_state)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        pass

    def _render_frame(self):
        pass

    def close(self):
        pass

# Main
if __name__ == "__main__":
    gym.register(
        id="tribes/Tribes-v0",
        entry_point=TribesEnv,
    )

    env = gym.make("tribes/Tribes-v0")
    env.reset()
    env = gym.make("tribes/Tribes-v0", size=10)
    env.reset()


