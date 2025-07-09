import torch
from torch import nn
import numpy as np

from beehavior.networks.dic_converter import deconcater


class BaseController(nn.Module):
    """
    control law, takes in observation and outputs a control vector
        observations can be tuple, tensor, or dict
        if tuple, assumes it is in order of keys in self.ksp, and handles accordingly
        if dict, assumes all keys in self.ksp are present, reforms into correctly ordered tuple, and reduces to first method
        if tensor, uses deconcater to split into
    keeps ksp=(keys, states, partitions), which keeps track of the order of the tensor fed in and how to split it
        splits using deconcater
    """

    def __init__(self, ksp, **kwargs):
        super().__init__(**kwargs)
        self.ksp = ksp

        if self.ksp is not None:
            self.keys, _, _ = self.ksp
            self.keys_to_idx = {k: i for i, k in enumerate(self.keys)}
        else:
            self.keys = None
            self.keys_to_idx = None

    def forward(self, x):
        if type(x) == tuple:
            pass
        elif type(x) == dict:
            x = tuple(x[k] for k in self.keys)
        else:
            # type(x) is tensor or np.array
            if len(x.shape) == 2:
                if type(x[0]) == np.ndarray:
                    return np.stack([self.forward(xp) for xp in x], axis=0)
                else:
                    return torch.stack([self.forward(xp) for xp in x], dim=0)
            x = deconcater(x, self.ksp)
        back_to_numpee = False
        if type(x[0]) == np.ndarray:
            back_to_numpee = True
            x = tuple(torch.tensor(arr, dtype=torch.float) for arr in x)
        output = self.forward_tuple(x)
        if back_to_numpee:
            output = output.detach().cpu().numpy()
        return output

    def forward_tuple(self, x):
        """
        takes in x, a tuple of tensors
        returns a control vector
        """
        raise NotImplementedError


class ProprotionalControl(BaseController):
    """
    keeps the OF magnitude sum at a constant c, keeps lateral OF at 0
    assumes observations are raw OF, OF x component, OF y component
    outputs a control vector proprtional to k1(c-(OF sum)), k2(sum(unnormalized x OF component))
    """

    def __init__(self,
                 c,
                 output_shape=(2,),
                 forward_idx=0,
                 sideways_idx=1,
                 proportions=(1/100, 2/100),
                 ksp=None,
                 divergence_kernel_size=3,
                 **kwargs
                 ):
        """
        Args:
            c: sum of magnitude should be kept at this value
            output_shape: shape of control output
            forward_idx: index to control forward acceleration
            sideways_idx: index to control rightwards acceleration
            proportions: proportions of control, default (1,1)
            ksp: keys, states, partition to unpack input tensor
            divergence_kernel_size: size to measure divegence
            **kwargs:
        """
        if ksp is None:
            default_shape = (3, 240, 320)
            ksp = (('front',),
                   (default_shape,),
                   (0, np.prod(default_shape)),
                   )
        super().__init__(ksp=ksp, **kwargs)
        self.c = c
        self.k1, self.k2 = proportions
        self.output_shape = output_shape
        self.fwd_idx = forward_idx
        self.side_idx = sideways_idx
        self.divergence_kernel = torch.zeros(2, divergence_kernel_size, divergence_kernel_size)
        for i in range(divergence_kernel_size):
            for j in range(divergence_kernel_size):
                x = j - (divergence_kernel_size - 1)/2
                y = -(i - (divergence_kernel_size - 1)/2)
                outwards_vec = torch.tensor([x, y])
                magnitude = torch.linalg.norm(outwards_vec)
                if magnitude > 0:
                    # normal vector to surface divided by magnitude, we want the resulting vector magnitude inverse proportional to surface of a circle at that radius
                    # in 2d, this is normal vector/(2pi r), proportional to normal vector/r
                    self.divergence_kernel[:, i, j] = (outwards_vec/magnitude)/(magnitude)
        # the sum of all magnitudes should be 1
        self.divergence_kernel = self.divergence_kernel/torch.sum(torch.linalg.norm(self.divergence_kernel, dim=0))
        self.divergence_kernel = self.divergence_kernel.unsqueeze(dim=0)

    def forward_tuple(self, x):
        OF = x[self.keys_to_idx['front']]
        OF_mag = OF[0]
        OF_x = OF_mag*OF[1]
        OF_y = OF_mag*OF[2]
        OF_vector_field = torch.stack((OF_x, OF_y), dim=0)  # (2, H, W)
        total_divergence = torch.sum(
            nn.functional.conv2d(input=OF_vector_field.unsqueeze(dim=0),
                                 weight=self.divergence_kernel)
        )
        mean_OF = torch.mean(OF_mag)
        lateral_mean = torch.mean(OF_x)

        _, H, W = OF.shape
        lateral_magnitude_diff = torch.mean(OF_mag[:, int(np.ceil(W/2)):]) - torch.mean(OF_mag[:, :int(np.floor(W/2))])

        output_vector = torch.zeros(self.output_shape)
        output_vector[self.fwd_idx] = self.k1*(self.c - mean_OF)
        output_vector[self.side_idx] = self.k2*(-lateral_magnitude_diff)
        return output_vector


if __name__ == '__main__':
    import os

    thingy = ProprotionalControl(c=33, divergence_kernel_size=3)
    print(thingy.divergence_kernel)
    print(thingy.divergence_kernel.shape)
    print(torch.sum(torch.linalg.norm(thingy.divergence_kernel, dim=1)))
    DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    output_dir = os.path.join(DIR, 'output', 'saved_control_laws')
    test_file = os.path.join(output_dir, 'prop_ctrl_c_' + str(thingy.c) + '.pkl')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    torch.save(thingy, test_file)

    thingy2 = torch.load(test_file)
    assert torch.equal(thingy.divergence_kernel, thingy2.divergence_kernel)
