from typing import Tuple, Optional
import gymnasium as gym
from gymnasium.core import ActType, ObsType
import numpy as np


class Test(gym.Env):
    """
    enviornemnt test
    state s is a coordinate in R2
    observation is a vector to a target coordinate in R2
    Actions are in [-1,1]^2, and T(s,a)=s+a
    reward of d(s,t)-d(s',t), rewarding getting close to target
    terminates if d(s,t)<1
    """

    def __init__(self, scale=10):
        super().__init__()
        self.scale = scale

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float64)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64)
        self.s = None
        self.t = None

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        d = np.linalg.norm(self.t - self.s)
        self.s += action
        dp = np.linalg.norm(self.t - self.s)
        r = d - dp

        # observation, reward, termination, truncation, info
        return self.get_obs(), r.item(), dp.item() < 1, False, {}

    def get_obs(self):
        return self.t - self.s

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        super().reset(seed=seed)
        np.random.seed(seed)
        self.s = (2*np.random.random(2) - 1)*self.scale
        self.t = (2*np.random.random(2) - 1)*self.scale
        # obs, info
        return self.get_obs(), {}


if __name__ == '__main__':
    env = Test()
    print(env.reset(seed=69))
    print(env.step(np.array([-1., 0])))
    print(env.step(np.array([1., 0])))
    print(env.step(np.array([1., -.25])))
