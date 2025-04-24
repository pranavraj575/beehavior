import numpy as np
import torch
from torch import nn
import ast
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def layer_from_config_dict(dic, input_shape=None, only_shape=False):
    """
    returns nn layer from a layer config dict
    handles Linear, flatten, relu, cnn, maxpool
    Args:
        dic: layer config dict
        {
            'type':type of layer (REQUIRED)
        }
        examples:
            {
             'type':'CNN',
             'channels':64,
             'kernel':(9,9),
             'stride':(3,3),
             'padding':(0,0),
            }

            {
             'type':'ReLU',
            }
        input_shape: BATCHED shape of input to network
            required for some layer types
            if we have unbatched_input_shape, then should set input_shape=(1,*unbatched_input_shape)
        only_shape: only calculate shapes, do not make networks
    Returns:
        layer, output shape
    """
    typ = dic['type'].lower()
    layer = None
    if typ == 'relu':
        if not only_shape: layer = nn.ReLU()
        shape = input_shape
    elif typ == 'flatten':
        start_dim = dic.get('start_dim', 1)
        end_dim = dic.get('end_dim', -1)
        if not only_shape:
            layer = nn.Flatten(start_dim=start_dim,
                               end_dim=end_dim,
                               )
        # INCLUSIVE of end dim
        if input_shape is not None:
            shape = (*(input_shape[:start_dim]),
                     np.prod(input_shape[start_dim:end_dim])*input_shape[end_dim],
                     *((input_shape[end_dim:])[1:]),  # do this to avoid issues with end_dim=-1
                     )
        else:
            shape = None

    elif typ == 'linear':
        out_features = dic['out_features']
        if not only_shape:
            layer = nn.Linear(in_features=input_shape[-1],
                              out_features=out_features,
                              )
        if input_shape is not None:
            shape = (*(input_shape[:-1]),
                     out_features,
                     )
        else:
            shape = None

    # image stuff has annoying output shape calculation
    # only need to write it once
    elif typ in ['cnn', 'maxpool']:
        (N, C, H, W) = input_shape
        kernel_size = dic['kernel']
        if type(kernel_size) == int: kernel_size = (kernel_size, kernel_size)
        stride = dic.get('stride', 1)
        if type(stride) == int: stride = (stride, stride)
        padding = dic.get('padding', 0)
        if type(padding) == int: padding = (padding, padding)

        Hp, Wp = ((H + 2*padding[0] - kernel_size[0])//stride[0] + 1,
                  (W + 2*padding[1] - kernel_size[1])//stride[1] + 1,
                  )
        if typ == 'cnn':
            out_channels = dic['channels']
            if not only_shape:
                layer = nn.Conv2d(
                    in_channels=C,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            shape = (N, out_channels, Hp, Wp)
        elif typ == 'maxpool':
            if not only_shape:
                layer = nn.MaxPool2d(
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )

            shape = (N, C, Hp, Wp)
        else:
            raise NotImplementedError
    else:
        raise Exception('type unknown:', typ)
    return layer, shape


def layers_from_config_list(dic_list, input_shape, only_shape=False):
    """
    returns list of layers from a list of config dicts
        calculates each successive input shape automatically
    Args:
        dic_list: list of layer config dicts
        input_shape: BATCHED shape of input to dict,
         probably required unless network is weird
        only_shape: only calculate shapes, do not make networks
    Returns:
        list of layers, output shape
    """
    layers = []
    shape = input_shape
    for dic in dic_list:
        layer, shape = layer_from_config_dict(dic=dic,
                                              input_shape=shape,
                                              only_shape=only_shape,
                                              )
        layers.append(layer)
    return layers, shape


def layers_from_config_file(file, input_shape=None, only_shape=False):
    """
    obtains layers from a file formatted as a python dict
    {
        'input_shape': input shape, OPTIONAL
        'layers': list of layer config dicts
    }
    Args:
        file: directory of file to read
        input_shape: overwrites input shape specified in file
        only_shape: only calculate shapes, do not make networks
    Returns:

    """
    f = open(file, 'r')
    full_dic = ast.literal_eval(f.read())
    f.close()
    if input_shape is None:
        input_shape = full_dic.get('input_shape', None)
    return layers_from_config_list(
        dic_list=full_dic['layers'],
        input_shape=input_shape,
        only_shape=only_shape,
    )


class CustomNN(BaseFeaturesExtractor):
    """
    custom network built with config file
    https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    """

    def __init__(self,
                 observation_space: gym.spaces.Box,
                 config_file,
                 ):
        unbatched = observation_space.shape
        _, output_shape = layers_from_config_file(file=config_file, input_shape=(1, *unbatched), only_shape=True)
        super().__init__(observation_space, output_shape[-1])
        layers, _ = layers_from_config_file(file=config_file, input_shape=(1, *unbatched))
        self.network = nn.Sequential(*layers)

        # Compute shape and print by doing one forward pass
        if True:
            with torch.no_grad():
                b = torch.as_tensor(observation_space.sample()[None]).float()
                print(b.shape)
                for layer in layers:
                    b = layer.forward(b)
                    print(b.shape, layer)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


if __name__ == '__main__':
    import os

    print(layer_from_config_dict(dic={'type': 'relu', }))
    print(layer_from_config_dict(dic={'type': 'flatten', 'start_dim': 2, 'end_dim': 4},
                                 input_shape=(1, 2, 3, 4, 5, 6),
                                 ))

    print(layer_from_config_dict(
        dic={'type': 'linear',
             'out_features': 64,
             },
        input_shape=(1, 128, 400, 400),
    ))

    print(layer_from_config_dict(
        dic={'type': 'CNN',
             'channels': 64,
             'kernel': (9, 8),
             'stride': (3, 2),
             'padding': (0, 1),
             },
        input_shape=(1, 128, 400, 400),
    ))
    print(layer_from_config_dict(
        dic={'type': 'maxpool',
             'kernel': (2, 2),
             'stride': (2, 2),
             },
        input_shape=(1, 128, 400, 400),
    ))
    network_dir = os.path.dirname(__file__)
    alex = os.path.join(network_dir, 'configs', 'alexnet.txt')
    print(layers_from_config_file(
        alex,
        input_shape=(1, 9, 240, 320)
    ))
    print()
    print(CustomNN(
        observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2, 240, 320)),
        config_file=alex,
    ))
