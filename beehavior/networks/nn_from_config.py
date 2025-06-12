import numpy as np
import torch
from torch import nn
import ast
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.fx.experimental.proxy_tensor import fetch_sym_proxy

from beehavior.networks.dic_converter import deconcater


def layer_from_config_dict(dic, input_shape=None, only_shape=False, device=None):
    """
    returns nn layer from a layer config dict
    handles Linear, flatten, relu, tanh, cnn, maxpool, avgpool, dropout, identity
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
    if typ == 'identity':
        if not only_shape: layer = nn.Identity()
        shape = input_shape
    elif typ == 'relu':
        if not only_shape: layer = nn.ReLU()
        shape = input_shape
    elif typ == 'tanh':
        if not only_shape: layer = nn.Tanh()
        shape = input_shape
    elif typ == 'dropout':
        if not only_shape: layer = nn.Dropout(dic.get('p', .1))
        shape = input_shape
    elif typ == 'dropout2d':
        if not only_shape: layer = nn.Dropout2d(dic.get('p', .1))
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
                              device=device,
                              )
        if input_shape is not None:
            shape = (*(input_shape[:-1]),
                     out_features,
                     )
        else:
            shape = None
    # image stuff has annoying output shape calculation
    # only need to write it once
    elif typ in ['cnn', 'maxpool', 'avgpool']:
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
                    device=device,
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
        elif typ == 'avgpool':
            if not only_shape:
                layer = nn.AvgPool2d(
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


def layers_from_config_list(dic_list, input_shape, only_shape=False, device=None):
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
                                              device=device,
                                              )
        layers.append(layer)
    return layers, shape


def layers_from_structure(structure, input_shape=None, only_shape=False, device=None):
    """
    obtains layers from a structure (dict, DO NOT NEST DICTS)
    {
        'input_shape': input shape, OPTIONAL
        'layers': list of layer config dicts
    }
    can also take dicts that contain these formatted things
        i.e. {'img':<nn b dict>, 'vec': <nn c dict>} will return
             {'img':<nn b layers>, 'vec': <nn c layers>},{'img':<nn b shape>, 'vec': <nn c shape>},
        IF you put in a dict, 'layers' cannot be a key, as this indicates the base level dict
        Can also do this:
            {'default':<nn a dict>, 'case a':'default', 'case b': default} will return
            {'default':<nn a layers>, 'case a':'default', 'case b': default},{'default':<nn a shape>, 'case a':'default', 'case b': default}
            In this case, only ONE network will be made, meant to handle two copies of identical input
            DO NOT PUT CHAINS LONGER THAN ONE,
    Args:
        structure: structure to read
        input_shape: overwrites input shape specified in file, MUST BE SAME FORM AS STRUCTURE
            i.e. if structure is a tuple, input shape must be tuple of input shapes, etc.
            also UNBATCHED
        only_shape: only calculate shapes, do not make networks
    Returns:
        layer structure, shape structure
    """
    if type(structure) == str:
        return structure, structure
    elif type(structure) == dict:
        if 'layers' in structure:
            if input_shape is None:
                input_shape = structure.get('input_shape', None)
            if input_shape is not None:
                input_shape = (1, *input_shape)
            return layers_from_config_list(
                dic_list=structure['layers'],
                input_shape=input_shape,
                only_shape=only_shape,
                device=device,
            )
        else:
            dd = {
                k: layers_from_structure(structure=structure[k],
                                         input_shape=None if input_shape is None else input_shape.get(k, None),
                                         only_shape=only_shape,
                                         )
                for k in structure
            }
            # layer dict, shape dict
            return {k: dd[k][0] for k in dd}, {k: dd[k][1] for k in dd}
    else:
        raise Exception('unknown type:', type(structure))


def layers_from_config_file(file, input_shape=None, only_shape=False):
    """
    obtains layers from a file formatted as a structure (calls layers_from_structure)
    Args:
        file: directory of file to read
        input_shape: overwrites input shape specified in file, UNBATCHED
        only_shape: only calculate shapes, do not make networks
    Returns:
    """
    f = open(file, 'r')
    full_file = ast.literal_eval(f.read())
    f.close()
    return layers_from_structure(structure=full_file, input_shape=input_shape, only_shape=only_shape)


class CustomNN(BaseFeaturesExtractor):
    """
    custom network built with config file
    https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    """

    def __init__(self,
                 observation_space,
                 structure,
                 ksp=None,
                 config_file=None,
                 ):
        """
        Args:
            observation_space:
            structure: specifies network structure
                can also put in None, then enter config file
            ksp: (keys, shapes, partition), order that INPUT is encoded
                if the input is a dictionary compressed into a tensor, this is how to uncompress it
                keys - order of keys in tensor
                shapes - shape of each element in tensor
                partition - what indices the tensor should be split at
            config_file:
        """
        self.dict_input = False
        self.ksp = ksp
        # ksp is the way INPUT is ordered
        # self.keys is the way OUTPUT is ordered
        #  then for the ith element of output, we will use self.network[self.keys[i]] on the self.keys_permutation[i]th element of concatenated INPUT
        self.keys_permutation = None  # self.keys[i]=keys[self.keys_permutation[i]] for keys,_,_= ksp
        unbatched = observation_space.shape

        if self.ksp is not None:
            keys, shapes, _ = self.ksp
            self.dict_input = True
            # unbatched needs to be swapped to a dict
            unbatched = {k: shapes[i] for i, k in enumerate(keys)}
        if unbatched is None:
            assert type(observation_space) == gym.spaces.Dict
            unbatched = {k: observation_space[k].shape for k in observation_space.keys()}
        if structure is None:
            assert config_file is not None
            f = open(config_file, 'r')
            structure = ast.literal_eval(f.read())
            f.close()
        _, output_shape = layers_from_structure(structure=structure, input_shape=unbatched, only_shape=True,
                                                device=None, )
        if type(output_shape) == dict:
            self.dict_input = True
            self.keys = tuple(sorted(output_shape.keys()))
            if self.ksp is not None:
                keys, _, _ = self.ksp
                self.keys_permutation = {i: keys.index(k) for i, k in enumerate(self.keys)}
            features_dim = 0
            for k in self.keys:
                if type(output_shape[k]) == str:
                    features_dim += output_shape[output_shape[k]][-1]
                else:
                    features_dim += output_shape[k][-1]
            super().__init__(observation_space, features_dim=features_dim)
        else:
            self.dict_input = False
            self.keys = None
            super().__init__(observation_space, output_shape[-1])
        layers, _ = layers_from_structure(structure=structure, input_shape=unbatched, device=None)
        if self.dict_input:
            network = {
                k: layers[k] if type(layers[k]) == str else nn.Sequential(*layers[k])
                for k in self.keys}
            network = {
                k: network[network[k]] if type(network[k]) == str else network[k]
                for k in self.keys
            }
            self.network = nn.ParameterDict(network)
        else:
            self.network = nn.Sequential(*layers)

        # Compute shape and print by doing one forward pass
        if True:
            with torch.no_grad():
                if self.dict_input:
                    for k in self.keys:
                        print('KEY:', k)
                        if self.ksp is None:
                            b = torch.as_tensor(observation_space[k].sample()[None], device=None).float()
                        else:
                            keys, shapes, partition = self.ksp
                            b = torch.as_tensor(observation_space.sample()[None], device=None).float()
                            idx = keys.index(k)
                            b = b[partition[idx]:partition[idx + 1]].reshape(shapes[idx])
                        lys = layers[k]
                        if type(lys) == str:
                            lys = layers[lys]
                        for layer in lys:
                            b = layer.forward(b)
                            print(b.shape, layer)
                else:
                    b = torch.as_tensor(observation_space.sample()[None]).float()
                    print(b.shape)
                    for layer in layers:
                        b = layer.forward(b)
                        print(b.shape, layer)

    def forward(self, observations):
        if self.dict_input:
            if type(observations) == dict:
                return torch.concatenate([self.network[k].forward(observations[k])
                                          for k in self.keys],
                                         dim=-1)
            else:
                stuff = deconcater(arr=observations, ksp=self.ksp)
                return torch.concatenate([self.network[k].forward(stuff[self.keys_permutation[i]])
                                          for i, k in enumerate(self.keys)],
                                         dim=-1,
                                         )
        else:
            return self.network.forward(observations)


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
        input_shape=(9, 240, 320)
    ))
    print()
    print(CustomNN(
        observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2, 240, 320)),
        config_file=alex,
        structure=None,
    ))

    alex = os.path.join(network_dir, 'configs', 'simple.txt')

    print(CustomNN(
        observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8, 240, 320)),
        config_file=alex,
        structure=None,
    ))
    f = open(alex, 'r')
    alex_struct = ast.literal_eval(f.read())
    f.close()
    f = open(os.path.join(network_dir, 'configs', 'simple.txt'), 'r')
    simplest_struct = ast.literal_eval(f.read())
    f.close()

    print(layers_from_structure(structure={'case a': alex_struct, 'case b': simplest_struct, 'case c': 'case a'},
                                input_shape={'case a': (8, 240, 320), 'case b': (8, 240, 320), 'case c': 'case a'},
                                only_shape=False,
                                ))
    obs_space = gym.spaces.Dict({'case a': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8, 240, 320)),
                                 'case b': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                                 'case c': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8, 240, 320))})
    pp = CustomNN(
        observation_space=
        obs_space,
        structure={
            'case a': simplest_struct,
            'case b': {
                'layers': [
                    {'type': 'linear', 'out_features': 256, },
                    {'type': 'ReLU', },
                ]
            },
            'case c': 'case a'
        },
    )
    print(pp)
    for param in pp.parameters():
        print(param.shape)
    # this should have more parameters, since we are making two copies of simplest_struct for case a and case c
    pp2 = CustomNN(
        observation_space=obs_space,
        structure={
            'case a': simplest_struct,
            'case b': {
                'layers': [
                    {'type': 'linear', 'out_features': 256, },
                    {'type': 'ReLU', },
                ]
            },
            'case c': simplest_struct
        },
    )

    print(pp2)
    for param in pp2.parameters():
        print(param.shape)
    print('second should be bigger:', len(list(pp.parameters())), len(list(pp2.parameters())))

    f = open(os.path.join(network_dir, 'configs', 'tiniest_hires_gc.txt'), 'r')
    gc_struct = ast.literal_eval(f.read())
    f.close()

    gc_space = gym.spaces.Dict({'front': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8, 240, 320)),
                                'vec': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                                'bottom': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8, 240, 320))})
    gc = CustomNN(
        observation_space=gc_space,
        structure=gc_struct,
    )
    print('params')
    for p in gc.parameters():
        print(p.shape)
