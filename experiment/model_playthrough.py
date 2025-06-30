import ast
import itertools
import shutil
import time

import gymnasium as gym
import stable_baselines3.common.policies
import torch.cuda

from beehavior.networks.nn_from_config import CustomNN


def pose_to_dic(pose):
    return {'position':
                np.array((pose.position.x_val,
                          pose.position.y_val,
                          pose.position.z_val,
                          )),
            'orientation':
                np.array((pose.orientation.x_val,
                          pose.orientation.y_val,
                          pose.orientation.z_val,
                          pose.orientation.w_val,
                          ))
            }


if __name__ == '__main__':

    import argparse
    import os

    DIR = os.path.dirname(os.path.dirname(__file__))

    PARSER = argparse.ArgumentParser(
        description='run RL on envionrment'
    )

    PARSER.add_argument("--models", nargs='+', required=True,
                        help="model file(s) (.pkl) to load. if multiple, uses first as default to gather trajectories, compares with rest (must be same input/output type)")

    PARSER.add_argument("--env-config-file", action='store', required=False, default=None,
                        help="environment specification file, formatted as dictionary {'name':<env name>,'kwargs':<kwargs dict>}")

    PARSER.add_argument("--output-dir-display", action='store', required=False, default=None,
                        help="directory to output saved info, defaults to output/<parent dir(parent dir(model path)>/display")

    PARSER.add_argument("--initial-pos", type=float, nargs=3, required=False, default=None,
                        help="initial position to spawn drone --initial-pos x y z")

    PARSER.add_argument("--capture-interval", type=int, required=False, default=3,
                        help="skip this many timesteps when displaying OF video")

    PARSER.add_argument("--num-trajectories", type=int, required=False, default=1,
                        help="number of trajectories to capture")

    PARSER.add_argument("--subsample-quiver", type=int, required=False, default=15,
                        help="how far to space quivers when displaying OF orientation")

    PARSER.add_argument("--baseline-amnt", type=float, required=False, default=None,
                        help="number of observations to use as background. "
                             "If None, uses all, "
                             "if less than 1, uses this as a proportion, "
                             "otherwise, uses this as number of samples")

    PARSER.add_argument("--reset", action='store_true', required=False,
                        help="overwrite previous trajectories")
    PARSER.add_argument("--append", action='store_true', required=False,
                        help="append sampled trajectory to previously captured trajectories")

    PARSER.add_argument("--avg-kernel", type=int, required=False, default=1,
                        help="smooth the attention with a moving average using this kernel (1 does nothing)")

    PARSER.add_argument("--range-to-display", type=int, nargs=2, required=False, default=[0, float('inf')],
                        help="display this range of frames")
    PARSER.add_argument("--max-over-each-frame", action='store_true', required=False,
                        help="take max over each frame for scaling instead of a global max over trajectories")
    PARSER.add_argument("--reexplain", action='store_true', required=False,
                        help="do not used saved explanations")
    PARSER.add_argument("--display-only", action='store_true', required=False,
                        help="only display route in simulation, do not save or analyze")
    PARSER.add_argument("--flip-axes", action='store_true', required=False,
                        help="flip any axes you are holding")
    PARSER.add_argument("--coolwarm-of", action='store_true', required=False,
                        help="optic flow in coolwarm map")
    PARSER.add_argument("--device", action='store', required=False, default='cpu',
                        help="device to store tensors on")
    args = PARSER.parse_args()

    import numpy as np
    from stable_baselines3 import PPO as MODEL
    import pickle as pkl
    import beehavior
    from beehavior.envs.goal_bee import GoalBee
    from experiment.trajectory_anal import create_mp4
    from experiment.shap_value_calc import shap_val, GymWrapper
    import cv2 as cv

    device = args.device
    output_dir = args.output_dir_display
    default_model_dir = args.models[0]
    if output_dir is None:
        output_dir = os.path.join(DIR,
                                  'output',
                                  os.path.basename(os.path.dirname(os.path.dirname(default_model_dir))),
                                  'display'
                                  )
    # traj_dir = os.path.join(output_dir, 'trajectories')
    # model_dir = os.path.join(output_dir, 'past_models')
    img_dir = os.path.join(output_dir, 'images')

    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    for d in (output_dir, img_dir):
        if not os.path.exists(d): os.makedirs(d)

    # load previous trajectory if found
    steps = None
    filename = os.path.join(output_dir, 'saved_traj.pkl')
    expln_filename = os.path.join(output_dir, 'saved_explanations.pkl')
    if os.path.exists(filename) and not args.reset:
        f = open(filename, 'rb')
        steps = pkl.load(f)
        f.close()
    # we are grabbing a trajectory if steps is empty, or if we are appending to steps, or displaying
    traj_sampling = ((steps is None) or args.append or args.display_only) and (args.num_trajectories > 0)

    if args.env_config_file is None:
        env_config = {'name': 'GoalBee-v0',
                      'kwargs': dict(
                          dt=.05,
                          action_type=GoalBee.ACTION_ACCELERATION,
                          img_history_steps=2,
                          input_velocity_with_noise=False,
                      )}
    else:
        f = open(args.env_config_file, 'r')
        env_config = ast.literal_eval(f.read())
        f.close()

    if not traj_sampling:
        # disable client
        env_config['kwargs']['client'] = False

    env = gym.make(env_config['name'],
                   **env_config['kwargs'],
                   )
    # so the order is 'front', 'bottom'
    of_camera_names = tuple(sorted(env.unwrapped.of_cameras, reverse=True))

    if not args.display_only:
        print('saving to', output_dir)

    if steps is None:
        steps = []

    if traj_sampling:
        model = MODEL.load(default_model_dir, env=env)
        model.policy.to(device)
        print(model.policy)
        for _ in range(args.num_trajectories):
            obs, info = env.reset(options={
                'initial_pos': args.initial_pos if args.initial_pos is not None else None
            })
            old_pose = env.unwrapped.get_pose()
            done = False
            while not done:
                action, _ = model.predict(observation=obs, deterministic=True)

                obs, rwd, done, term, info = env.step(action)
                pose = env.unwrapped.get_pose()
                if env.unwrapped.concat_obs:
                    dic_obs = env.unwrapped.obs_to_dict(obs)
                else:
                    dic_obs = obs
                images = env.unwrapped.get_rgb_imgs()
                steps.append({
                    'old_pose': pose_to_dic(old_pose),
                    'rgb_imgs': images,
                    'obs': obs,
                    'dic_obs': dic_obs,
                    'action': action,
                    'reward': rwd,
                    'pose': pose_to_dic(pose),
                    'info': info,
                })
                old_pose = pose
            print(info)
        if args.display_only:
            quit()
        print('completed sample, saving')
        time.sleep(4)
        print('saving for real')
        time.sleep(1)
        # if old explanations exist, remove them
        if os.path.exists(expln_filename):
            os.remove(expln_filename)
        f = open(filename, 'wb')
        pkl.dump(steps, f)
        f.close()
        print('saved')

    print('timesteps:', len(steps))

    # calculate explanations
    explanations = dict()
    if os.path.exists(expln_filename) and not args.reexplain:
        # since every time we edit sampled steps, we delete this file,
        #   this file only will exist if we have correct info in it
        f = open(expln_filename, 'rb')
        explanations = pkl.load(f)
        f.close()
    expln_keys = args.models

    updated_exp = False
    for expln_key in expln_keys:
        if expln_key not in explanations:
            updated_exp = True
            print('explainin', expln_key)
            model = MODEL.load(expln_key, env=env)
            model.policy.to(device)

            converted_observations = [
                model.policy.obs_to_tensor(dic['obs'])[0] for dic in steps
            ]
            if type(converted_observations[0]) == dict:
                raise Exception("dict conversion does not work with shap explainer")
            else:
                tensor_observations = torch.concatenate(converted_observations, dim=0)
                wrapped_model = GymWrapper(network=model.policy,
                                           model_call_kwargs={'deterministic': True},
                                           )
            # sample a proportion of the data for background
            if args.baseline_amnt is not None:
                if args.baseline_amnt < 1:
                    # baseline_amnt is a proportion
                    idxs = torch.randperm(len(tensor_observations))[:int(len(tensor_observations)*args.baseline_amnt)]
                else:
                    # baseline_amnt is a number
                    idxs = torch.randperm(len(tensor_observations))[:int(args.baseline_amnt)]
            else:
                # idxs = torch.arange(len(tensor_observations))
                idxs = torch.randperm(len(tensor_observations))
            baseline_tensor = tensor_observations[idxs]
            explanations[expln_key] = shap_val(
                model=wrapped_model,
                explanation_data=tensor_observations,
                baseline=baseline_tensor,
                progress=True,
            )

    if updated_exp:
        print('obtained explanations, saving them')
        f = open(expln_filename, 'wb')
        pkl.dump(explanations, f)
        f.close()
        print('saved')
    # all_abs_explanations[explain (model) key][output key] gives a list of explanations
    # if type of each explanations[expln_key] is a list, there are multiple output keys, 'sum', 0, 1, ...
    # otherwise, there is only one ouput key, 'sum'

    all_abs_explanations = {
        expln_key: {
            'sum': [sum(np.abs(ex) for ex in expln) if type(expln) is list else np.abs(expln)
                    for expln in explanations[expln_key]
                    ],
            **{
                i: [np.abs(ex[i]) for ex in explanations[expln_key]] for i in range(len(explanations[expln_key][0]))
                if type(explanations[expln_key][0]) is list
            },
        }
        for expln_key in explanations
    }
    output_keys = list(all_abs_explanations[expln_keys[0]].keys())
    total_timesteps = len(all_abs_explanations[expln_keys[0]][output_keys[0]])

    # take avg across comparison models
    all_abs_explanations['avg'] = {
        output_key: [
            sum(all_abs_explanations[expln_key][output_key][t] for expln_key in expln_keys)/len(expln_keys)
            for t in range(total_timesteps)
        ]
        for output_key in output_keys
    }
    expln_keys = ['avg'] + expln_keys
    all_abs_explanations = {
        expln_key: {
            output_key: [env.unwrapped.obs_to_dict(expln) for expln in explaination_list]
            for output_key, explaination_list in output_key_to_list.items()
        }
        for expln_key, output_key_to_list in all_abs_explanations.items()
    }

    # all_abs_explanations[explain(model) key][output key] is a list with explanations
    # each explanation is in the same shape as environmnet observations (usually dictionary(camera key -> tensor))

    # in this, each explanation is restricted to OF types at current timesteps
    # explanation is a dictionary (camera key -> (OF key -> tensor))
    #  OF keys are every possible input type, as well as a sum over all input types
    current_img_abs_explanations = dict()
    obs_type_keys = set()
    for expln_key, output_key_to_list in all_abs_explanations.items():
        current_img_abs_explanations[expln_key] = dict()
        for output_key, explaination_list in output_key_to_list.items():
            current_img_abs_explanations[expln_key][output_key] = []
            for expln in explaination_list:
                thing = dict()
                for cam_name in of_camera_names:
                    camera_obs_tensor = expln[cam_name]
                    camera_obs_tensor = camera_obs_tensor.reshape(camera_obs_tensor.shape[-3:])
                    thing[cam_name] = dict()
                    i = len(camera_obs_tensor)
                    for obs_type_key in env.unwrapped.ordered_input_img_space[::-1]:
                        if obs_type_key == GoalBee.INPUT_OF_ORIENTATION:
                            i -= 2
                            for k, dimkey in enumerate(('x', 'y')):
                                thing[cam_name][(obs_type_key, dimkey)] = camera_obs_tensor[i + k]
                                obs_type_keys.add((obs_type_key, dimkey))
                        else:
                            i -= 1
                            thing[cam_name][obs_type_key] = camera_obs_tensor[i]
                            obs_type_keys.add(obs_type_key)
                    thing[cam_name]['sum'] = sum(ex for obs_type_key, ex in thing[cam_name].items())
                    obs_type_keys.add('sum')
                current_img_abs_explanations[expln_key][output_key].append(thing)
    obs_type_keys = list(obs_type_keys)
    # episode_maxes[explain(model) key][output key][obs_type_key] is a max over
    # current_img_abs_explanations[explain(model) key][output key][i][camera key][obs_type_key] for i in all displayed time, and all camera key
    #  IF output_key is not 'sum', takes max over all non-sum output keys as well
    #  IF obs_type_key is not 'sum', takes max over all non-sum obs_type_keys as well
    # if max over each frame:
    # episode_maxes_per_frame[explain(model) key][output key][i][obs_type_key]

    blurring_kernel = cv.getGaussianKernel(ksize=args.avg_kernel, sigma=args.avg_kernel/2)
    blurring_kernel = blurring_kernel.dot(blurring_kernel.T)
    blurred_episode_maxes = dict()
    blurred_episode_maxes_per_frm = dict()
    for expln_key, output_key_to_list in current_img_abs_explanations.items():
        blurred_episode_maxes[expln_key] = dict()
        blurred_episode_maxes_per_frm[expln_key] = dict()
        for output_key in output_keys:
            if output_key == 'sum':
                output_key_iter = [output_key]
            else:
                output_key_iter = [ok for ok in output_keys if ok != 'sum']
                if output_key_iter[0] in blurred_episode_maxes[expln_key]:
                    # the max is the same for all output keys in output_key_iter
                    blurred_episode_maxes[expln_key][output_key] = blurred_episode_maxes[expln_key][output_key_iter[0]]
                    blurred_episode_maxes_per_frm[expln_key][output_key] = \
                        blurred_episode_maxes_per_frm[expln_key][output_key_iter[0]]
                    continue
            blurred_episode_maxes[expln_key][output_key] = dict()
            blurred_episode_maxes_per_frm[expln_key][output_key] = [dict() for _ in range(total_timesteps)]

            for obs_type_key in obs_type_keys:
                if obs_type_key == 'sum':
                    obs_type_key_iter = [obs_type_key]
                else:
                    obs_type_key_iter = [otk for otk in obs_type_keys if otk != 'sum']
                    if obs_type_key_iter[0] in blurred_episode_maxes[expln_key][output_key]:
                        blurred_episode_maxes[expln_key][output_key][obs_type_key] = \
                            blurred_episode_maxes[expln_key][output_key][obs_type_key_iter[0]]
                        if args.max_over_each_frame:
                            for t in range(total_timesteps):
                                blurred_episode_maxes_per_frm[expln_key][output_key][t][obs_type_key] = \
                                    blurred_episode_maxes_per_frm[expln_key][output_key][t][obs_type_key_iter[0]]
                        continue
                blurred_episode_maxes[expln_key][output_key][obs_type_key] = max(
                    max(
                        max(
                            max(
                                np.max(cv.filter2D(obs_key_to_expln[otk], -1, blurring_kernel))
                                for otk in obs_type_key_iter
                            )
                            for _, obs_key_to_expln in cam_key_obs_key_to_expln.items()
                        )
                        for cam_key_obs_key_to_expln in output_key_to_list[ok]
                    )
                    for ok in output_key_iter
                )
                if args.max_over_each_frame:
                    for t in range(total_timesteps):
                        blurred_episode_maxes_per_frm[expln_key][output_key][t][obs_type_key] = max(
                            max(
                                max(
                                    np.max(cv.filter2D(obs_key_to_expln[otk], -1, blurring_kernel))
                                    for otk in obs_type_key_iter
                                )
                                for _, obs_key_to_expln in output_key_to_list[ok][t].items()
                            )
                            for ok in output_key_iter
                        )

    # display optic flow
    assert ((GoalBee.INPUT_LOG_OF in env.unwrapped.input_img_space) or
            (GoalBee.INPUT_RAW_OF in env.unwrapped.input_img_space))

    OF_scale = dict()

    count = int((GoalBee.INPUT_INV_DEPTH_IMG in env.unwrapped.input_img_space) +
                2*(GoalBee.INPUT_OF_ORIENTATION in env.unwrapped.input_img_space)
                )

    if GoalBee.INPUT_OF_ORIENTATION in env.unwrapped.input_img_space:
        for cam_name in of_camera_names:
            OF_scale[cam_name] = np.mean([
                np.max(
                    np.exp(dic['dic_obs'][cam_name][-count - 1:][0])
                    if GoalBee.INPUT_LOG_OF in env.unwrapped.input_img_space
                    else dic['dic_obs'][cam_name][-count - 1:][0]
                )
                for dic in steps]
            )

    import matplotlib.pyplot as plt


    def make_plot(t,
                  cam_name,
                  expln_key,
                  obs_type_key='sum',
                  plot_OF=True,
                  quiver_plot=False,
                  output_key=None,
                  plotter=None,
                  filename=None,
                  ):
        """
        plots and saves optic flow info and attention
        Args:
            t: timestep to get dictionary from
            cam_name: name of OF camera
            plot_OF: use OF as the image instead of RGB view
            output_key: output key of attention to plot ('sum', 0, ..., n-1, None) where n is the output dimension
              if None, does not plot attention
            plotter: plt object
            filename: filename to save to
        Returns:
        """
        if plotter is None:
            plotter = plt
        dic = steps[t]

        # get optic flow information
        dic_obs = dic['dic_obs']
        OF = dict()
        stack_bottom = 0
        cmap = None
        for input_k in env.unwrapped.ordered_input_img_space[::-1]:
            stack_size = 2 if input_k == GoalBee.INPUT_OF_ORIENTATION else 1
            stack_bottom = stack_bottom - stack_size
            im = dic_obs[cam_name]
            OF[input_k] = im[len(im) + stack_bottom:len(im) + stack_bottom + stack_size]
            if stack_size == 1:
                OF[input_k] = OF[input_k][0]

        OF_magnitude = (OF[GoalBee.INPUT_RAW_OF] if GoalBee.INPUT_RAW_OF in OF
                        else np.exp(OF[GoalBee.INPUT_LOG_OF]))
        OF_log_magnitude = (OF[GoalBee.INPUT_LOG_OF] if GoalBee.INPUT_LOG_OF in OF
                            else np.log(np.clip(OF[GoalBee.INPUT_RAW_OF], 10e-3, np.inf)))

        if plot_OF:
            # make an OF image
            if args.coolwarm_of:
                # not making quivers, use heatmap
                img = 255*OF_magnitude/np.max(OF_magnitude)  # np.stack([OF_magnitude for _ in range(3)], axis=-1)
                cmap = 'coolwarm'
            else:
                # making quivers, so make a black and white image
                log_mag_min = np.min(OF_log_magnitude)
                log_mag_max = np.max(OF_log_magnitude)
                img = (np.stack([OF_log_magnitude for _ in range(3)], axis=-1) - log_mag_min)
                if log_mag_max == log_mag_min:
                    img = np.zeros_like(img)
                else:
                    img = img/(log_mag_max - log_mag_min)
                # scale image to pixels
                img = img*255
        else:
            # use the rgb image
            img = dic['rgb_imgs'][cam_name][:, :, ::-1]

        # if we want to display attention
        attention = None
        if output_key is not None:
            # img = .5*img  # mute the OF information
            # change the green channel to attention
            # current_img_abs_explanations[explain(model) key][output key][i][camera key][obs_type_key]
            expln = current_img_abs_explanations[expln_key][output_key][t][cam_name][obs_type_key]
            # cam_name_to_obs_type_to_expln = current_img_abs_explanations[expln_key][output_key][t]

            blurred_attention = cv.filter2D(expln, -1, blurring_kernel)

            if args.max_over_each_frame:
                scale = blurred_episode_maxes_per_frm[expln_key][output_key][t][obs_type_key]
            else:
                scale = blurred_episode_maxes[expln_key][output_key][obs_type_key]
                # episode_maxes[explain(model) key][output key][obs_type_key]
            attention = np.clip(blurred_attention, 0, np.inf)/scale
            if plot_OF:  # need to split the channels
                cmap = None
                if len(img.shape) == 2:
                    img = np.stack([img for _ in range(3)], axis=-1)  # todo : make this better
                img[:, :, 1] = attention*255  # scaled to [0,255]
        if (output_key is not None) and (not plot_OF):  # only attention is being plotted, can use a heatmap
            plotter.imshow(attention,
                           cmap='coolwarm',
                           interpolation='nearest',
                           vmin=0,
                           vmax=1,
                           )
        else:
            # either we are plotting attention alongside OF, in which case we split channels
            #  or we are plotting visual info, in which case img is the visual image
            img = np.ndarray.astype(img, dtype=np.uint8)
            plotter.imshow(img, interpolation='nearest', cmap=cmap)

        if quiver_plot:
            # quiver of OF
            ss = args.subsample_quiver
            h, w = np.meshgrid(np.arange(OF_log_magnitude.shape[0]), np.arange(OF_log_magnitude.shape[1]))
            OF_orientation = OF[GoalBee.INPUT_OF_ORIENTATION]
            OF_magnitude = np.exp(OF_log_magnitude)
            of_disp = np.transpose(OF_orientation*np.expand_dims(OF_magnitude, 0),
                                   axes=(0, 2, 1))
            # inverted from image (height is top down) to np plot (y dim  bottom up)

            plotter.quiver(w[::ss, ::ss], h[::ss, ::ss],
                           of_disp[0, ::ss, ::ss],
                           -of_disp[1, ::ss, ::ss],
                           color='black' if args.coolwarm_of else 'red',
                           scale=OF_scale[cam_name]*(max(w.shape)/ss),
                           # width=.005,
                           )
        plotter.set_xticklabels([])
        plotter.set_xticks([])

        plotter.set_yticklabels([])
        plotter.set_yticks([])
        # plotter.show()
        if filename is not None:
            plotter.savefig(filename, bbox_inches='tight')


    all_settings = []

    # all cameras in the x axis
    # in the y axis, (visual, optic flow, attention)
    all_settings.append(
        (sum(
            [[
                {
                    'plt coords': (0, i),
                    'output_key': None,
                    'quiver_plot': False,
                    'plot_OF': False,
                    'cam_name': cam_name,
                },
                {
                    'plt coords': (1, i),
                    'output_key': None,
                    'quiver_plot': True,
                    'plot_OF': True,
                    'cam_name': cam_name,
                },
                {
                    'plt coords': (2, i),
                    'output_key': 'sum',
                    'quiver_plot': False,
                    'plot_OF': False,
                    'cam_name': cam_name,
                },
            ] for i, cam_name in enumerate(of_camera_names)], []
        ),
         {
             'ident': 'all',
             'subplot_dim': (3, len(of_camera_names)),
             'xlabels': of_camera_names,
             'ylabels': ['visual',
                         'optic flow',
                         'attention'
                         ],
             'xlabel_kwargs': {'fontsize': 20},
             'ylabel_kwargs': {'fontsize': 20},
             'flip_axes': args.flip_axes,
         },
        )
    )

    # split previous into each row
    for single_plt in ('visual', 'optic flow', 'attention'):
        all_settings.append(
            (
                [
                    {
                        'plt coords': (0, i),
                        'output_key': 'sum' if single_plt == 'attention' else None,
                        'quiver_plot': single_plt == 'optic flow',
                        'plot_OF': single_plt == 'optic flow',
                        'cam_name': cam_name,
                    }
                    for i, cam_name in enumerate(of_camera_names)],
                {
                    'ident': single_plt.replace(' ', '_'),
                    'subplot_dim': (1, len(of_camera_names)),
                    'xlabels': of_camera_names if len(of_camera_names) > 1 else None,
                    'xlabel_kwargs': {'fontsize': 20},
                    'flip_axes': args.flip_axes,
                    'vid': False,
                },
            )
        )
        # split OF into individual plots
        if single_plt == 'optic flow' and len(of_camera_names) > 1:
            for i, cam_name in enumerate(of_camera_names):
                all_settings.append(
                    (
                        [
                            {
                                'plt coords': (0, 0),
                                'output_key': None,
                                'quiver_plot': True,
                                'plot_OF': True,
                                'cam_name': cam_name,
                            }
                        ],
                        {
                            'ident': 'optic_flow_' + cam_name,
                            'subplot_dim': (1, 1),
                            'vid': False,
                        },
                    )
                )

    # comparison of different models for each camera
    # x axis is (avg, individual model 1, ... )
    # y axis is (visual, optic flow, attention)
    # visual and optic flow only are shown for one model, since the rest are duplicates
    # also a version with y axis being cameras
    if len(expln_keys) > 2:  # if expln keys is not just [avg][model]
        for cam_name in of_camera_names:
            all_settings.append(
                (sum(
                    [[
                         {
                             'plt coords': (0, i),
                             'output_key': None,
                             'quiver_plot': False,
                             'plot_OF': False,
                             'cam_name': cam_name,
                             'expln_key': expln_key,
                         },
                         {
                             'plt coords': (1, i),
                             'output_key': None,
                             'quiver_plot': True,
                             'plot_OF': True,
                             'cam_name': cam_name,
                             'expln_key': expln_key,
                         },
                         {
                             'plt coords': (2, i),
                             'output_key': 'sum',
                             'quiver_plot': False,
                             'plot_OF': False,
                             'cam_name': cam_name,
                             'expln_key': expln_key,
                         },
                     ][0 if i == 0 else -1:]  # if i>0, first two plots are redundant
                     for i, expln_key in enumerate(expln_keys)
                     ],
                    []
                ),
                 {
                     'subplot_dim': (3, len(expln_keys)),
                     'ident': 'comp_vis_' + cam_name,
                     'xlabels': ['average'] + ['model ' + str(i) for i in (range(len(expln_keys) - 1))],
                     'ylabels': ['visual', 'optic flow', 'attention'],
                     'xlabel_kwargs': {'fontsize': 20},
                     'ylabel_kwargs': {'fontsize': 20},
                     'flip_axes': args.flip_axes,
                 }
                )
            )
        # cameras in y axis
        # explanation model in x axis
        all_settings.append((
            sum(
                [
                    [
                        {
                            'plt coords': (i, j),
                            'output_key': 'sum',
                            'quiver_plot': False,
                            'plot_OF': False,
                            'cam_name': cam_name,
                            'expln_key': expln_key,
                        }
                        for i, cam_name in enumerate(of_camera_names)]
                    for j, expln_key in enumerate(expln_keys)
                ],
                []
            ),
            {
                'subplot_dim': (len(of_camera_names), len(expln_keys)),
                'ident': 'comp',
                'xlabels': ['Average attention'] + ['Model ' + str(i) for i in (range(len(expln_keys) - 1))],
                'ylabels': of_camera_names if len(of_camera_names) > 1 else None,
                'xlabel_kwargs': {'fontsize': 30},
                'ylabel_kwargs': {'fontsize': 30},
                'flip_axes': args.flip_axes,
                'vid': False,
            }
        )
        )
        # split the previous up by explain key
        # dont need this
        for j, expln_key in enumerate(expln_keys):
            continue
            name = (['average'] + ['model ' + str(i) for i in (range(len(expln_keys) - 1))])[j]
            all_settings.append((
                [
                    {
                        'plt coords': (i, 0),
                        'output_key': 'sum',
                        'quiver_plot': False,
                        'plot_OF': False,
                        'cam_name': cam_name,
                        'expln_key': expln_key,
                    }
                    for i, cam_name in enumerate(of_camera_names)
                ],
                {
                    'subplot_dim': (len(of_camera_names), 1),
                    'ident': 'comp_' + name.replace(' ', '_'),
                    'ylabels': of_camera_names if len(of_camera_names) > 1 else None,
                    'xlabel_kwargs': {'fontsize': 20},
                    'ylabel_kwargs': {'fontsize': 20},
                    'flip_axes': args.flip_axes,
                    'vid': False,
                }
            )
            )

    # comparison of attention for each output key
    # cameras in y axis
    # output keys in x axis
    all_settings.append((
        sum(
            [[
                {
                    'plt coords': (j, i),
                    'output_key': output_key,
                    'quiver_plot': False,
                    'plot_OF': True,
                    'cam_name': cam_name,
                }
                for j, output_key in enumerate(output_keys)
            ] for i, cam_name in enumerate(of_camera_names)], []
        ),
        {
            'ident': 'keys',
            'subplot_dim': (len(output_keys), len(of_camera_names)),
            'xlabels': of_camera_names if len(of_camera_names) > 1 else None,
            'ylabels': output_keys,
            'xlabel_kwargs': {'fontsize': 20},
            'ylabel_kwargs': {'fontsize': 20},
            'flip_axes': args.flip_axes,
        },
    )
    )
    # for each camera
    # visual, OF, attention in y axis
    # each obs_type_key in x axis
    for cam_name in of_camera_names:
        all_settings.append(
            (sum(
                [[
                     {
                         'plt coords': (0, i),
                         'output_key': None,
                         'quiver_plot': False,
                         'plot_OF': False,
                         'cam_name': cam_name,
                         'obs_type_key': otk,
                     },
                     {
                         'plt coords': (1, i),
                         'output_key': None,
                         'quiver_plot': True,
                         'plot_OF': True,
                         'cam_name': cam_name,
                         'obs_type_key': otk,
                     },
                     {
                         'plt coords': (2, i),
                         'output_key': 'sum',
                         'quiver_plot': False,
                         'plot_OF': False,
                         'cam_name': cam_name,
                         'obs_type_key': otk,
                     },
                 ][0 if i == 0 else -1:]  # if i>0, first two plots are redundant
                 for i, otk in enumerate(filter(lambda x: x != 'sum', obs_type_keys))
                 ],
                []
            ),
             {
                 'subplot_dim': (3, len(obs_type_keys) - 1),
                 'ident': 'obs_layer_comp_' + cam_name,
                 'xlabels': list(filter(lambda x: x != 'sum', obs_type_keys)),
                 'ylabels': ['visual', 'optic flow', 'attention'],
                 'xlabel_kwargs': {'fontsize': 20},
                 'ylabel_kwargs': {'fontsize': 20},
                 'flip_axes': args.flip_axes,
                 'vid': True,
             }
            )
        )
    # for each camera
    # plot obs type keys in x
    # output key in y
    #  i.e. for each observation type, which output does it influence most
    for cam_name in of_camera_names:
        all_settings.append(
            (sum(
                [[
                    {
                        'plt coords': (i, j),
                        'output_key': output_key,
                        'quiver_plot': False,
                        'plot_OF': False,
                        'cam_name': cam_name,
                        'obs_type_key': otk,
                    }
                    for i, output_key in enumerate(filter(lambda x: x != 'sum', output_keys))
                ]
                    for j, otk in enumerate(filter(lambda x: x != 'sum', obs_type_keys))
                ],
                []
            ),
             {
                 'subplot_dim': (len(output_keys) - 1, len(obs_type_keys) - 1),
                 'ident': 'key_to_output_' + cam_name,
                 'xlabels': list(filter(lambda x: x != 'sum', obs_type_keys)),
                 'ylabels': list(filter(lambda x: x != 'sum', output_keys)),
                 'xlabel_kwargs': {'fontsize': 20},
                 'ylabel_kwargs': {'fontsize': 20},
                 'flip_axes': args.flip_axes,
                 'vid': False,
             }
            )
        )
    # make plots
    for settings in all_settings:
        ident = None
        subplot_dim = None
        if type(settings) == dict:
            settings = [settings]
            info = dict(ident='')
        else:
            settings, info = settings
            ident = info['ident']
            subplot_dim = info['subplot_dim']
            flip_axes = info.get('flip_axes', False)
            if flip_axes:
                subplot_dim = subplot_dim[::-1]
                info['xlabels'], info['ylabels'] = info.get('ylabels', None), info.get('xlabels', None)
        print('collecting frames for', ident)
        img_files = []
        for t, dic in enumerate(steps):
            low_rng, high_rng = args.range_to_display
            if (t - low_rng)%args.capture_interval:
                # capture every 'capture_interval'th frame starting at low_rng
                continue
            if t < low_rng or t >= high_rng:
                # ouf of display range
                continue
            if subplot_dim is not None:
                fig, axs = plt.subplots(subplot_dim[0], subplot_dim[1])
                if type(axs) != np.ndarray:
                    axs = np.array(axs)
                axs = axs.reshape(subplot_dim)
                for ax in axs.flatten():
                    ax.set_xticklabels([])
                    ax.set_xticks([])

                    ax.set_yticklabels([])
                    ax.set_yticks([])
                w, h = fig.get_size_inches()
                fig.set_size_inches(w*subplot_dim[1], h*subplot_dim[0])
                plt.subplots_adjust(
                    # left=0.125, right=0.9,
                    # bottom=0.1, top=0.9,
                    wspace=0.0, hspace=0.0
                )

                if 'xlabels' in info and info['xlabels'] is not None:
                    for xlabel, ax in zip(info['xlabels'], axs[-1, :]):
                        ax.set_xlabel(xlabel, **info.get('xlabel_kwargs', dict()))
                if 'ylabels' in info and info['ylabels'] is not None:
                    for ylabel, ax in zip(info['ylabels'], axs[:, 0]):
                        ax.set_ylabel(ylabel, **info.get('ylabel_kwargs', dict()))
                plotters = [axs[stuff['plt coords'][::-1 if flip_axes else 1]] for stuff in settings]
            else:
                plotters = [plt.gca()]

            for plotter, stuff in zip(plotters, settings):
                output_key = stuff['output_key']
                plot_OF = stuff['plot_OF']
                cam_name = stuff['cam_name']
                expln_key = stuff.get('expln_key', 'avg')
                obs_type_key = stuff.get('obs_type_key', 'sum')
                if ident is None:
                    ident = stuff['ident']
                if 'quiver_plot' in stuff:
                    quiver_plot = stuff['quiver_plot']
                else:
                    quiver_plot = ((output_key is None) and
                                   (GoalBee.INPUT_OF_ORIENTATION in env.unwrapped.input_img_space)
                                   )
                make_plot(t=t,
                          expln_key=expln_key,
                          cam_name=cam_name,
                          obs_type_key=obs_type_key,
                          output_key=output_key,
                          plot_OF=plot_OF,
                          plotter=plotter,
                          quiver_plot=quiver_plot,
                          filename=None,
                          )
            filename = os.path.join(img_dir,
                                    ident + '_' + str(t) +
                                    '.png'
                                    )
            plt.savefig(filename, bbox_inches='tight')
            img_files.append(filename)
            plt.close()
        if info.get('vid', True):
            filename = os.path.join(output_dir,
                                    ident + '_' +
                                    'OF.mp4')
            print('made frames, forming video:', filename)
            create_mp4(image_paths=img_files,
                       output_mp4_path=filename,
                       duration=env.unwrapped.dt*1000*args.capture_interval,
                       debug=True,
                       )

    quit()

    plt.plot([dic['action'][0] for dic in steps], label='x command')
    plt.plot([(dic['pose']['position'][0] - dic['old_pose']['position'][0])/env.unwrapped.dt for dic in steps],
             label='x velocity')
    plt.legend()
    plt.show()

    plt.plot([dic['action'][1] for dic in steps], label='y command')
    plt.plot([(dic['pose']['position'][1] - dic['old_pose']['position'][1])/env.unwrapped.dt for dic in steps],
             label='y velocity')
    plt.legend()
    plt.show()
