import os.path

import airsim

from airsim_interface.load_settings import get_fov
from beehavior.envs.forward_bee import ForwardBee
import gymnasium as gym
import numpy as np
from airsim_interface.interface import of_geo, get_depth_img, CAMERA_SETTINGS
import matplotlib.pyplot as plt

env_config = {'name': 'ForwardBee-v0',
              'kwargs': dict(

                  input_img_space=(ForwardBee.INPUT_RAW_OF,
                                   ForwardBee.INPUT_OF_ORIENTATION,
                                   ),
                  action_type=ForwardBee.ACTION_ACCELERATION_XY,
                  img_history_steps=1,
                  concatenate_observations=False,
              )}

env = gym.make(env_config['name'],
               **env_config['kwargs'],
               )

env.reset(options={'initial_pos': (0, 37.69, -3),
                   'wiggle': True,
                   })
depth_map = get_depth_img(client=env.unwrapped.client,
                          camera_name='front',
                          numpee=True,
                          )
fwd_v = np.array([1., 0., 0.])
fwd_v = fwd_v/np.linalg.norm(fwd_v)

right_v = np.array([1., 1., 0.])
right_v = right_v/np.linalg.norm(right_v)

forward_OF = of_geo(depth_map=depth_map,
                    linear_velocity=fwd_v,
                    angular_velocity=np.zeros(3),
                    FOVx_degrees=get_fov(camera_settings=CAMERA_SETTINGS,
                                         camera_name='front',
                                         image_type=airsim.ImageType.DepthPerspective,
                                         )
                    )
rightwards_OF = of_geo(depth_map=depth_map,
                       linear_velocity=right_v,
                       angular_velocity=np.zeros(3),
                       FOVx_degrees=get_fov(camera_settings=CAMERA_SETTINGS,
                                            camera_name='front',
                                            image_type=airsim.ImageType.DepthPerspective,
                                            )
                       )
min_val = min([np.min(np.linalg.norm(OF, axis=0)) for OF in (forward_OF, rightwards_OF)])
min_val = 0
max_val = max([np.max(np.linalg.norm(OF, axis=0)) for OF in (forward_OF, rightwards_OF)])


def plt_OF(OF, title=None, save=None):
    plt.imshow(np.linalg.norm(OF, axis=0), interpolation='nearest', cmap='coolwarm',
               vmin=min_val,
               vmax=max_val,
               )
    if title is not None:
        plt.title(title)
    ss = 15
    h, w = np.meshgrid(np.arange(OF.shape[1]), np.arange(OF.shape[2]))

    of_disp = np.transpose(OF,
                           axes=(0, 2, 1))
    # inverted from image (height is top down) to np plot (y dim  bottom up)

    plt.quiver(w[::ss, ::ss], h[::ss, ::ss],
               of_disp[0, ::ss, ::ss],
               -of_disp[1, ::ss, ::ss],
               color='black',
               scale=.5*max_val*max(w.shape)/ss,
               # width=.005,
               )
    plt.yticks([0, 60, 120, 180, 240],
               labels=['$60^\\circ$',
                       '$30^\\circ$',
                       '$0^\\circ$',
                       '$-30^\\circ$',
                       '$-60^\\circ$'])

    plt.xticks([0, 80, 160, 240, 320],
               labels=['$-60^\\circ$',
                       '$-30^\\circ$',
                       '$0^\\circ$',
                       '$30^\\circ$',
                       '$60^\\circ$'])
    if save is not None:
        plt.savefig(save, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'output')
plt_OF(forward_OF,
       title='velocity ' + str(fwd_v) + ', max |OF| is ' + str(round(np.max(np.linalg.norm(forward_OF, axis=0)))),
       save=os.path.join(output_dir, 'fwd_of.png')),
plt_OF(rightwards_OF,
       title='velocity ' + str(right_v*np.sqrt(2)) + '/$\\sqrt{2}$, max |OF| is ' + str(
           round(np.max(np.linalg.norm(rightwards_OF, axis=0)))),
       save=os.path.join(output_dir, 'fwd_right_of.png'),
       )


def equatorial_OF(OF):
    mag = np.linalg.norm(OF, axis=0)
    equator = mag[int(np.floor(mag.shape[0]/2)):int(np.ceil(mag.shape[0]/2)) + 1, :]
    return np.mean(equator, axis=0)


plt.plot(np.linspace(-60, 60, forward_OF.shape[-1]),
         equatorial_OF(forward_OF),
         label='drone with velocity ' + str(fwd_v),
         color='blue')
plt.xlabel('angle in degrees')
plt.ylabel('equatorial OF magnitude')
plt.plot(np.linspace(-60, 60, rightwards_OF.shape[-1]),
         equatorial_OF(rightwards_OF),
         label='drone with velocity ' + str(right_v*np.sqrt(2)) + '/$\\sqrt{2}$',
         color='orange',
         )
ylim = plt.ylim()

mindex_fwd = np.argmin(np.linalg.norm(forward_OF, axis=0))
mindex_fwd = np.unravel_index(mindex_fwd, forward_OF.shape[1:])
mindex_rgt = np.argmin(np.linalg.norm(rightwards_OF, axis=0))
mindex_rgt = np.unravel_index(mindex_rgt, rightwards_OF.shape[1:])

angle_fwd = np.linspace(-60, 60, forward_OF.shape[-1])[mindex_fwd[1]]
angle_rgt = np.linspace(-60, 60, rightwards_OF.shape[-1])[mindex_rgt[1]]
plt.plot([angle_fwd, angle_fwd], ylim, label='focus of expansion', color='blue', alpha=.5, linestyle='--')
plt.plot([angle_rgt, angle_rgt], ylim, label='focus of expansion', color='orange', alpha=.5, linestyle='--')

plt.ylim(ylim)
plt.xlabel('angle in degrees')
plt.ylabel('equatorial OF magnitude')
plt.title('Equatorial OF')
plt.legend()
plt.savefig(os.path.join(output_dir, 'of_equatorial.png'))

print(mindex_fwd, mindex_rgt)
