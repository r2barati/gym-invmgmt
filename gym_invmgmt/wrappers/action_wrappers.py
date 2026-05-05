"""
Action wrappers for inventory management environments.
"""

import gymnasium as gym
import numpy as np


class IntegerActionWrapper(gym.ActionWrapper):
    """
    Rounds continuous actions to integers before passing to the environment.

    Supply chain order quantities are physically discrete — you can't order
    3.7 units. This wrapper converts any float action to the nearest integer
    while preserving the continuous action space (so RL gradient flows are
    unaffected).

    Works with both array and dict action formats.
    """

    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        if isinstance(action, dict):
            return {k: np.round(v) for k, v in action.items()}
        return np.round(action)
