from typing import Tuple, Optional
import gymnasium as gym
from gymnasium.core import ActType, ObsType
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
import torch


class Test(gym.Env):
    """
    enviornemnt test
    state s is a coordinate in R2
    observation is a vector to a target coordinate in R2, as well as random 'images'
    Actions are in [-1,1]^2, and T(s,a)=s+a
    reward of d(s,t)-d(s',t), rewarding getting close to target
    terminates if d(s,t)<1
    """

    def __init__(self, scale=100):
        super().__init__()
        self.scale = scale

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float64)
        self.observation_space = gym.spaces.Dict({
            'vec': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64),
            'img': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2, 2, 2), dtype=np.float64),
            'img2': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2, 2, 2), dtype=np.float64),
        })
        self.s = None
        self.t = None

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        """
        implementiaon of gym's step function
        Args:
            action: action that agent makes at a timestep
        Returns:
            (observation, reward, termination,
                    truncation, info)
        """
        d = np.linalg.norm(self.t - self.s)
        self.s += action
        dp = np.linalg.norm(self.t - self.s)
        r = d - dp
        self.ct += 1

        # observation, reward, termination, truncation, info
        return self.get_obs(), r.item(), dp.item() < 1 or self.ct > 1000, False, {}

    def get_obs(self):
        vec = self.t - self.s
        img = self.observation_space['img'].sample()
        img2 = self.observation_space['img2'].sample()
        return {
            'vec': vec,
            'img': img,
            'img2': img2,
        }

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        """
        implementiaon of gym's reset function
        Args:
            seed: random seed
            options: option dictionary
        Returns:
            (observation, info dict)
        """
        super().reset(seed=seed)
        self.ct = 0
        np.random.seed(seed)
        self.s = (2*np.random.random(2) - 1)*self.scale
        self.t = (2*np.random.random(2) - 1)*self.scale
        # obs, info
        return self.get_obs(), {}


class TestNN(BaseFeaturesExtractor):
    """
    custom network built with config file
    https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    """

    def __init__(self,
                 observation_space,
                 ):
        """
        Args:
            observation_space:
            structure: specifies network structure
                can also put in None, then enter config file
            config_file:
        """
        unbatched = observation_space.shape
        if unbatched is None:
            assert type(observation_space) == gym.spaces.Dict
            self.keys = tuple(sorted(observation_space.keys()))

        features_dim = 16*3
        super().__init__(observation_space, features_dim=features_dim)
        img_net = nn.Sequential(nn.Flatten(), nn.Linear(in_features=8, out_features=16), nn.ReLU())
        self.network = nn.ModuleDict({
            'vec': nn.Sequential(nn.Linear(in_features=2, out_features=16), nn.ReLU()),
            'img': img_net,
            'img2': img_net,
        })

    def forward(self, observations):
        stuff = []
        for k in self.keys:
            obs = observations[k]
            stuff.append(self.network[k].forward(obs))
        return torch.concatenate(stuff, dim=-1)


class Test2(gym.Env):
    """testing the concatenated thingy
    """

    def __init__(self, scale=100):
        super().__init__()
        self.scale = scale

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float64)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64)

        self.s = None
        self.t = None

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        """
        implementiaon of gym's step function
        Args:
            action: action that agent makes at a timestep
        Returns:
            (observation, reward, termination,
                    truncation, info)
        """
        d = np.linalg.norm(self.t - self.s)
        self.s += action
        dp = np.linalg.norm(self.t - self.s)
        r = d - dp
        self.ct += 1

        # observation, reward, termination, truncation, info
        return self.get_obs(), r.item(), dp.item() < 1 or self.ct > 1000, False, {}

    def get_obs(self):
        vec = self.t - self.s
        img = np.random.random((2, 2, 2))
        img2 = np.random.random((2, 2, 2))
        dic = {
            'vec': vec,
            'img': img,
            'img2': img2,
        }
        return np.concatenate((dic['img'].flatten(), dic['img2'].flatten(), dic['vec']))

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        """
        implementiaon of gym's reset function
        Args:
            seed: random seed
            options: option dictionary
        Returns:
            (observation, info dict)
        """
        super().reset(seed=seed)
        self.ct = 0
        np.random.seed(seed)
        self.s = (2*np.random.random(2) - 1)*self.scale
        self.t = (2*np.random.random(2) - 1)*self.scale
        # obs, info
        return self.get_obs(), {}


class TestNN2(BaseFeaturesExtractor):
    """
    custom network built with config file
    https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    """

    def __init__(self,
                 observation_space,
                 ):
        """
        Args:
            observation_space:
            structure: specifies network structure
                can also put in None, then enter config file
            config_file:
        """
        unbatched = observation_space.shape
        if unbatched is None:
            assert type(observation_space) == gym.spaces.Dict
            self.keys = tuple(sorted(observation_space.keys()))

        features_dim = 16*3
        super().__init__(observation_space, features_dim=features_dim)
        img_net = nn.Sequential(nn.Flatten(), nn.Linear(in_features=8, out_features=16), nn.ReLU())
        self.network = nn.ModuleDict({
            'vec': nn.Sequential(nn.Linear(in_features=2, out_features=16), nn.ReLU()),
            'img': img_net,
            'img2': img_net,
        })

    def forward(self, observations):
        if len(observations.shape) == 1:
            observations = observations.reshape((1, -1))
        img = observations[:, :8].reshape((-1, 2, 2, 2))
        img2 = observations[:, 8:16].reshape((-1, 2, 2, 2))
        vec = observations[:, 16:]
        print('shps')
        print(self.network['img'].forward(img).shape)
        print(self.network['vec'].forward(vec).shape)
        return torch.concatenate((self.network['img'].forward(img),
                                  self.network['img2'].forward(img2),
                                  self.network['vec'].forward(vec)),
                                 dim=-1
                                 )


if __name__ == '__main__':
    env = Test()
    print(env.observation_space.sample())
    print(type(env.observation_space))
    print(env.observation_space['img'].shape)
    print(env.reset(seed=69))
    print(env.step(np.array([-1., 0])))
    print(env.step(np.array([1., 0])))
    print(env.step(np.array([1., -.25])))

    import gymnasium as gym
    import os

    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    import matplotlib.pyplot as plt

    steps_per_epoch = 256

    policy_kwargs = dict(
        features_extractor_class=TestNN,
    )
    model = PPO("MultiInputPolicy", env, verbose=0, n_steps=steps_per_epoch, policy_kwargs=policy_kwargs, device='cpu')
    for _ in range(100):
        model.learn(total_timesteps=steps_per_epoch, reset_num_timesteps=False, progress_bar=False)
        obs, info = env.reset()  # vec_env.reset()
        term = False
        reward_all = []
        while not term:
            action, _states = model.predict(obs)
            obs, rewards, term, _, info = env.step(action)
            reward_all.append(rewards)
        print(sum(reward_all)/len(reward_all))
