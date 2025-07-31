"""
CNN feature extractor
This is outdated, replaced by nn_from_config.py
"""
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
    :param maxpools: sequence of whether to maxpool at each layer
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self,
                 observation_space: gym.spaces.Box,
                 channels=(32, 64),
                 kernels=(8, 4),
                 strides=(4, 2),
                 paddings=(0, 0),
                 ffn_hidden_layers=(),
                 features_dim: int = 256,
                 maxpools=True,
                 ):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        layers = []
        chanl_in = n_input_channels
        if type(maxpools) == bool:
            maxpools = (maxpools for _ in channels)
        for chanl_out, kernel, stride, padding, maxpool in zip(channels, kernels, strides, paddings, maxpools):
            layers.append(nn.Conv2d(chanl_in,
                                    chanl_out,
                                    kernel_size=kernel,
                                    stride=stride,
                                    padding=padding,
                                    )
                          )
            layers.append(nn.ReLU())
            if maxpool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            chanl_in = chanl_out
        layers.append(nn.Flatten())

        self.cnn = nn.Sequential(*layers)

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

            if True:  # print out shapes
                b = torch.as_tensor(observation_space.sample()[None]).float()
                print(b.shape)
                for layer in layers:
                    b = layer.forward(b)
                    print(b.shape, layer)
        temp = n_flatten
        layers_ffn = []
        for hidden in ffn_hidden_layers:
            layers_ffn.append(nn.Linear(temp, hidden))
            layers_ffn.append(nn.ReLU())
            temp = hidden
        layers_ffn.append(nn.Linear(temp, features_dim))
        layers_ffn.append(nn.ReLU())
        self.linear = nn.Sequential(*layers_ffn)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


if __name__ == '__main__':
    cnn = CNN(observation_space=gym.spaces.Box(-float('inf'), float('inf'), (3, 100, 100)))
    print(cnn.forward(observations=torch.rand((23, 3, 100, 100))).shape)
