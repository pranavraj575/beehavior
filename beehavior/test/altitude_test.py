import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


class CustomCNN(BaseFeaturesExtractor):
    """
    https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


if __name__ == '__main__':
    import beehavior
    from stable_baselines3 import PPO as MODEL

    env = gym.make('HiBee-v0', dt=.1, )

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )

    model = MODEL('CnnPolicy', env, verbose=1, policy_kwargs=policy_kwargs,
                  # buffer_size=1000,  # for replay buffer methods
                  n_steps=128,
                  )
    model.learn(total_timesteps=20000)
    for _ in range(100):
        obs, _ = env.reset()
        rwds = []
        done = False
        while not done:
            action, _ = model.predict(observation=obs, deterministic=False)
            obs, rwd, done, term, info = env.step(action)
            rwds.append(rwd)
        print('ep length:', len(rwds))
        print('rwd mean:', sum(rwds)/len(rwds))
