import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


class CNN(BaseFeaturesExtractor):
    """
    https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    :param observation_space: (gym.Space)
    :param channels: sequence of channels to map image into
    :param kernels: sequence of kernel sizes
    :param strides: sequence of strides
    :param paddings: sequence of paddings
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self,
                 observation_space: gym.spaces.Box,
                 channels=(32, 64),
                 kernels=(8, 4),
                 strides=(4, 2),
                 paddings=(0, 0),
                 features_dim: int = 256,
                 ):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        layers = []
        chanl_in = n_input_channels
        for chanl_out, kernel, stride, padding in zip(channels, kernels, strides, paddings):
            layers.append(nn.Conv2d(chanl_in,
                                    chanl_out,
                                    kernel_size=kernel,
                                    stride=stride,
                                    padding=padding,
                                    )
                          )
            layers.append(nn.ReLU())
            chanl_in=chanl_out
        layers.append(nn.Flatten())

        self.cnn = nn.Sequential(*layers)

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
    cnn = CNN(observation_space=gym.spaces.Box(-float('inf'), float('inf'), (3, 100, 100)))
    print(cnn.forward(observations=torch.rand((23,3,100,100))).shape)