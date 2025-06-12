import ast
import itertools
import shutil

import gymnasium as gym
import stable_baselines3.common.policies
import torch.cuda

from beehavior.networks.nn_from_config import CustomNN

if __name__ == '__main__':

    import argparse
    import os

    DIR = os.path.dirname(os.path.dirname(__file__))

    PARSER = argparse.ArgumentParser(
        description='run RL on envionrment'
    )

    # PARSER.add_argument("--env", action='store', required=False, default='HiBee-v0',
    #                    choices=('Beese-v0', 'HiBee-v0', 'ForwardBee-v0'),
    #                   help="RL gym class to run")

    PARSER.add_argument("--model", action='store', required=True,
                        help="model file (.pkl) to load")

    PARSER.add_argument("--env-config-file", action='store', required=False, default=None,
                        help="environment specification file, formatted as dictionary {'name':<env name>,'kwargs':<kwargs dict>}")

    PARSER.add_argument("--display-output-dir", action='store', required=False, default=None,
                        help="directory to output saved info, defaults to output/<parent dir(parent dir(model path)>/display")

    PARSER.add_argument("--initial-pos", type=float, nargs=3, required=False, default=None,
                        help="initial position to spawn drone --initial-pos x y z")

    PARSER.add_argument("--capture-interval", type=int, required=False, default=3,
                        help="skip this many timesteps when displaying OF video")
    PARSER.add_argument("--ignorientation", action='store_true', required=False,
                        help="ignore OF orientation")

    PARSER.add_argument("--num-trajectories", type=int, required=False, default=1,
                        help="number of trajectories to capture")

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
    PARSER.add_argument("--reexplain", action='store_true', required=False,
                        help="do not used saved explanations")
    PARSER.add_argument("--display-only", action='store_true', required=False,
                        help="only display route in simulation, do not save or analyze")
    PARSER.add_argument("--device", action='store', required=False, default='cpu',
                        help="device to store tensors on")
    args = PARSER.parse_args()

    import numpy as np
    from stable_baselines3 import PPO as MODEL
    import pickle as pkl
    import beehavior
    from beehavior.envs.goal_bee import GoalBee
    from experiment.trajectory_anal import create_gif
    from experiment.shap_value_calc import shap_val, GymWrapper
    import cv2 as cv

    device = args.device
    output_dir = args.display_output_dir
    if output_dir is None:
        output_dir = os.path.join(DIR,
                                  'output',
                                  os.path.basename(os.path.dirname(os.path.dirname(args.model))),
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

    model = MODEL.load(args.model, env=env)
    model.policy.to(device)
    print(model.policy)
    print(type(model.policy))
    if not args.display_only:
        print('saving to', output_dir)


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


    if steps is None:
        steps = []

    if traj_sampling:
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
        if args.display_only:
            quit()
        print('completed sample, saving')
        # if old explanations exist, remove them
        if os.path.exists(expln_filename):
            os.remove(expln_filename)
        f = open(filename, 'wb')
        pkl.dump(steps, f)
        f.close()
        print('saved')

    import matplotlib.pyplot as plt

    # calculate explanations
    explanations = None
    if os.path.exists(expln_filename) and not args.reexplain:
        # since every time we edit sampled steps, we delete this file,
        #   this file only will exist if we have correct info in it
        f = open(expln_filename, 'rb')
        explanations = pkl.load(f)
        f.close()
    else:
        converted_observations = [
            model.policy.obs_to_tensor(dic['obs'])[0] for dic in steps
        ]
        if type(converted_observations[0]) == dict:
            raise Exception("dict conversion does not work with shap explainer")
            # tensor_observations, ksp = dict_to_tensor(dic=converted_observations)
            # wrapped_model = DicWrapper(model.policy,
            #                           ksp=ksp,
            #                           proc_model_output=lambda x: x[0].flatten(),
            #                           model_call_kwargs={'deterministic': True},
            #                           )
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
        explanations = shap_val(model=wrapped_model,
                                explanation_data=tensor_observations,
                                baseline=baseline_tensor,
                                progress=True,
                                )
        print('obtained explanations, saving them')
        f = open(expln_filename, 'wb')
        pkl.dump(explanations, f)
        f.close()
        print('saved')

    all_abs_explanations = [
        {'sum': sum(np.abs(ex) for ex in expln),
         **{i: np.abs(ex) for i, ex in enumerate(expln)}}
        if type(expln) is list
        else {'sum': np.abs(expln)}
        for expln in explanations
    ]
    all_abs_explanations = [
        {
            output: env.unwrapped.obs_to_dict(explnation_for_each_output[output])
            for output in explnation_for_each_output
        }
        for explnation_for_each_output in all_abs_explanations
    ]

    # for each timestep t, all_abs_explanations[t] is a dictionary dic with
    #   dic[i]-> abs(explanation of output i)
    #   dic['sum']-> sum_i(abs(explanation of output i))
    # each explanation is in the same shape as environmnet observations

    # current_sum_all_abs_explanations looks at only the current timestep, and takes the sum to make a
    #  'image' the same size as each optic flow input
    current_sum_all_abs_explanations = []
    for explnation_for_each_output in all_abs_explanations:
        dic = dict()
        for output_key in explnation_for_each_output:
            explanation = explnation_for_each_output[output_key]
            current_obs_dic = {
                obs_key:
                    explanation[obs_key].reshape(-1, *explanation[obs_key].shape[-2:])[-env.unwrapped.imgs_per_step:]
                for obs_key in env.unwrapped.of_cameras}
            current_sum_obs_dic = {
                obs_key: np.sum(current_obs_dic[obs_key], axis=0)
                for obs_key in env.unwrapped.of_cameras
            }
            dic[output_key] = current_sum_obs_dic
        current_sum_all_abs_explanations.append(dic)

    output_keys = list(current_sum_all_abs_explanations[0].keys())
    # max_current_sum_all_abs_explanations[output_key] gives the max across all time steps, across all cameras
    #  of shap values with respect to a particular output key
    max_current_sum_all_abs_explanations = {
        output_key: max(
            max(
                np.max(dic[output_key][cam_name])
                for dic in current_sum_all_abs_explanations
            )
            for cam_name in env.unwrapped.of_cameras
        )
        for output_key in output_keys
    }
    blurring_kernel = cv.getGaussianKernel(ksize=args.avg_kernel, sigma=args.avg_kernel/2)
    blurring_kernel = blurring_kernel.dot(blurring_kernel.T)
    blurred_max_current_sum_all_abs_explanations = {
        output_key: max(
            max(
                np.max(cv.filter2D(dic[output_key][cam_name], -1, blurring_kernel))
                for dic in current_sum_all_abs_explanations
            )
            for cam_name in env.unwrapped.of_cameras
        )
        for output_key in output_keys
    }

    # display optic flow
    assert ((GoalBee.INPUT_LOG_OF in env.unwrapped.input_img_space) or
            (GoalBee.INPUT_RAW_OF in env.unwrapped.input_img_space))

    OF_scale = dict()

    count = int((GoalBee.INPUT_INV_DEPTH_IMG in env.unwrapped.input_img_space) +
                2*(GoalBee.INPUT_OF_ORIENTATION in env.unwrapped.input_img_space)
                )

    if GoalBee.INPUT_OF_ORIENTATION in env.unwrapped.input_img_space:
        for cam_name in env.unwrapped.of_cameras:
            OF_scale[cam_name] = np.mean([
                np.max(
                    np.exp(dic['dic_obs'][cam_name][-count - 1:][0])
                    if GoalBee.INPUT_LOG_OF in env.unwrapped.input_img_space
                    else dic['dic_obs'][cam_name][-count - 1:][0]
                )
                for dic in steps]
            )


    def make_plot(t,
                  cam_name,
                  use_OF=True,
                  quiver_plot=False,
                  output_key=None,
                  only_heatmap=False,
                  plotter=None,
                  filename=None,
                  ):
        """
        plots and saves optic flow info and attention
        Args:
            t: timestep to get dictionary from
            cam_name: name of OF camera
            use_OF: use OF as the image instead of RGB view
            output_key: output key to plot ('sum', 0, ..., n-1, None) where n is the output dimension
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
        for input_k in env.unwrapped.ordered_input_img_space[::-1]:
            stack_size = 2 if input_k == GoalBee.INPUT_OF_ORIENTATION else 1
            stack_bottom = stack_bottom - stack_size
            im = dic_obs[cam_name]
            OF[input_k] = im[len(im) + stack_bottom:len(im) + stack_bottom + stack_size]
            if stack_size == 1:
                OF[input_k] = OF[input_k][0]
        OF_log_magnitude = (OF[GoalBee.INPUT_LOG_OF] if GoalBee.INPUT_LOG_OF in OF
                            else np.log(np.clip(OF[GoalBee.INPUT_RAW_OF], 10e-3, np.inf)))
        if use_OF:
            # make an OF image
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
            csae = current_sum_all_abs_explanations[t][output_key][cam_name]

            blurred_csae = cv.filter2D(csae, -1, blurring_kernel)
            attention = (blurred_csae/np.max(blurred_csae))  # scaled to [0,1]
            # attention = np.clip(blurred_csae, 0, np.inf)/blurred_max_current_sum_all_abs_explanations[output_key]
            if not only_heatmap:
                img[:, :, 1] = attention*255  # scaled to [0,255]
        if only_heatmap:
            plotter.imshow(attention,
                           cmap='coolwarm',
                           interpolation='nearest',
                           vmin=0,
                           vmax=1,
                           )
        else:
            img = np.ndarray.astype(img, dtype=np.uint8)
            plotter.imshow(img, interpolation='nearest', )

        if quiver_plot:
            ss = 10
            h, w = np.meshgrid(np.arange(OF_log_magnitude.shape[0]), np.arange(OF_log_magnitude.shape[1]))
            OF_orientation = OF[GoalBee.INPUT_OF_ORIENTATION]
            OF_magnitude = np.exp(OF_log_magnitude)
            of_disp = np.transpose(OF_orientation*np.expand_dims(OF_magnitude, 0),
                                   axes=(0, 2, 1))
            # inverted from image (height is top down) to np plot (y dim  bottom up)

            plotter.quiver(w[::ss, ::ss], h[::ss, ::ss],
                           of_disp[0, ::ss, ::ss],
                           -of_disp[1, ::ss, ::ss],
                           color='red',
                           scale=OF_scale[cam_name]*(max(w.shape)/ss),
                           )
        plotter.set_xticklabels([])
        plotter.set_xticks([])

        plotter.set_yticklabels([])
        plotter.set_yticks([])
        # plotter.show()
        if filename is not None:
            plotter.savefig(filename, bbox_inches='tight')


    all_settings = []
    for output_key in output_keys + [None]:
        for use_OF, only_heatmap in itertools.product((True, False), repeat=2):
            if output_key is not None and not use_OF:
                # is we are using the rgb image, we do not need to define output_key
                continue
            if output_key is None and only_heatmap:
                # only use heatmaps when output key is defined
                continue
            for cam_name in env.unwrapped.of_cameras:
                sett = {
                    'output_key': output_key,
                    'use_OF': use_OF,
                    'only_heatmap': only_heatmap,
                    'cam_name': cam_name,
                    'ident': (str(output_key) + '_' if output_key is not None else '') +
                             ('heat_' if only_heatmap else '') +
                             ('of_' if use_OF else 'raw_') + cam_name
                }
                all_settings.append(sett)
    all_settings.append(
        ((3, 1),
         [
             {
                 'plt coords': (0, 0),
                 'output_key': None,
                 'quiver_plot': False,
                 'use_OF': False,
                 'only_heatmap': False,
                 'cam_name': 'front',
             },
             {
                 'plt coords': (1, 0),
                 'output_key': None,
                 'quiver_plot': True,
                 'use_OF': True,
                 'only_heatmap': False,
                 'cam_name': 'front',
             },
             {
                 'plt coords': (2, 0),
                 'output_key': 'sum',
                 'quiver_plot': False,
                 'use_OF': False,
                 'only_heatmap': True,
                 'cam_name': 'front',
             },
         ],
         'all'
         )
    )
    all_settings = all_settings[::-1]
    # make plots for all output keys, all timesteps, all cameras
    for settings in all_settings:
        ident = None
        subplot_dim = None
        if type(settings) == dict:
            settings = [settings]
        else:
            subplot_dim, settings, ident = settings
        img_files = []
        for t, dic in enumerate(steps):
            if t%args.capture_interval:
                continue
            if subplot_dim is not None:
                fig, axs = plt.subplots(subplot_dim[0], subplot_dim[1])
                w, h = fig.get_size_inches()
                fig.set_size_inches(w*subplot_dim[1], h*subplot_dim[0])
                plt.subplots_adjust(
                    # left=0.125, right=0.9,
                    # bottom=0.1, top=0.9,
                    wspace=0.0, hspace=0.0
                )
                if any(p == 1 for p in subplot_dim):
                    # ignore the 0 and just index once
                    plotters = [axs[max(stuff['plt coords'])] for stuff in settings]
                else:
                    plotters = [axs[stuff['plt coords']] for stuff in settings]
            else:
                plotters = [plt.gca()]

            for plotter, stuff in zip(plotters, settings):
                output_key = stuff['output_key']
                only_heatmap = stuff['only_heatmap']
                use_OF = stuff['use_OF']
                cam_name = stuff['cam_name']
                if ident is None:
                    ident = stuff['ident']
                if 'quiver_plot' in stuff:
                    quiver_plot = stuff['quiver_plot']
                else:
                    quiver_plot = ((output_key is None) and
                                   (GoalBee.INPUT_OF_ORIENTATION in env.unwrapped.input_img_space) and
                                   (not args.ignorientation)
                                   )
                make_plot(t=t,
                          cam_name=cam_name,
                          output_key=output_key,
                          use_OF=use_OF,
                          plotter=plotter,
                          only_heatmap=only_heatmap,
                          quiver_plot=quiver_plot,
                          filename=None,
                          )
            filename = os.path.join(img_dir,
                                    ident + '_' + str(t) +
                                    '.png'
                                    )
            plt.savefig(filename)
            img_files.append(filename)
            plt.close()
        filename = os.path.join(output_dir,
                                ident + '_' +
                                'OF_gifed.gif')
        create_gif(image_paths=img_files,
                   output_gif_path=filename,
                   duration=env.unwrapped.dt*1000*args.capture_interval,
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
