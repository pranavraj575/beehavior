import ast
import shutil

import gymnasium as gym
import stable_baselines3.common.policies
import torch.cuda

from beehavior.networks.nn_from_config import CustomNN

if __name__ == '__main__':

    import argparse
    import numpy as np
    from stable_baselines3 import PPO as MODEL
    import os
    import pickle as pkl
    import beehavior
    from beehavior.envs.goal_bee import GoalBee
    from experiment.trajectory_anal import create_gif

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
    PARSER.add_argument("--clear-imgs", action='store_true', required=False,
                        help="clear any previous imgs")
    PARSER.add_argument("--retry", action='store_true', required=False,
                        help="take another trajectory")

    args = PARSER.parse_args()
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

    if args.clear_imgs and os.path.exists(img_dir):
        shutil.rmtree(img_dir)

    for d in (output_dir, img_dir):
        if not os.path.exists(d): os.makedirs(d)

    # load previous trajectory if found
    steps = None
    filename = os.path.join(output_dir, 'saved_traj.pkl')
    if os.path.exists(filename) and not args.retry:
        f = open(filename, 'rb')
        steps = pkl.load(f)
        f.close()

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

    if steps is not None:
        # disable client
        env_config['kwargs']['client'] = False
    env = gym.make(env_config['name'],
                   **env_config['kwargs'],
                   )

    model = MODEL.load(args.model, env=env)

    print(model.policy)
    print(type(model.policy))
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
        obs, info = env.reset(options={
            'initial_pos': args.initial_pos if args.initial_pos is not None else None
        })
        old_pose = env.unwrapped.get_pose()
        done = False
        while not done:
            action, _ = model.predict(observation=obs, deterministic=True)

            obs, rwd, done, term, info = env.step(action)
            pose = env.unwrapped.get_pose()

            steps.append({
                'old_pose': pose_to_dic(old_pose),
                'obs': obs,
                'action': action,
                'reward': rwd,
                'pose': pose_to_dic(pose),
                'info': info,
            })
            old_pose = pose

        f = open(filename, 'wb')
        pkl.dump(steps, f)
        f.close()

    import matplotlib.pyplot as plt

    capture_interval = args.capture_interval

    # display optic flow
    if ((GoalBee.INPUT_LOG_OF in env.unwrapped.input_img_space) or
            (GoalBee.INPUT_RAW_OF in env.unwrapped.input_img_space)):

        of_img_files = {cam: [] for cam in env.unwrapped.of_cameras}
        OF_scale = {cam: 1 for cam in env.unwrapped.of_cameras}

        count = int((GoalBee.INPUT_INV_DEPTH_IMG in env.unwrapped.input_img_space) +
                    2*(GoalBee.INPUT_OF_ORIENTATION in env.unwrapped.input_img_space)
                    )

        if GoalBee.INPUT_OF_ORIENTATION in env.unwrapped.input_img_space:
            for cam_name in env.unwrapped.of_cameras:
                OF_scale[cam_name] = np.mean([
                    np.max(
                        np.exp(dic['obs'][cam_name][-count - 1:][0])
                        if GoalBee.INPUT_LOG_OF in env.unwrapped.input_img_space
                        else dic['obs'][cam_name][-count - 1:][0]
                    )
                    for dic in steps]
                )

        for t, dic in enumerate(steps):
            if t%capture_interval:
                continue
            obs = dic['obs']
            np_action, _ = model.predict(obs, deterministic=True)
            obs_tense, vectorized = model.policy.obs_to_tensor(obs)
            actions, value, log_prob = model.policy.forward(obs_tense, deterministic=True)
            print(np_action, actions.cpu().detach().numpy())
            print(np_action-actions.cpu().detach().numpy())
            print()
            # print(actions, value, log_prob)
            for cam_name in env.unwrapped.of_cameras:
                OF = dict()
                i = 0
                for input_k in env.unwrapped.ordered_input_img_space[::-1]:
                    width = 2 if input_k == GoalBee.INPUT_OF_ORIENTATION else 1
                    i = i - width

                    im = obs[cam_name]
                    OF[input_k] = im[len(im) + i:len(im) + i + width]
                    if width == 1:
                        OF[input_k] = OF[input_k][0]

                OF_log_magnitude = (OF[GoalBee.INPUT_LOG_OF] if GoalBee.INPUT_LOG_OF in OF
                                    else np.log(np.clip(OF[GoalBee.INPUT_RAW_OF], 10e-3, np.inf)))

                log_mag_min = np.min(OF_log_magnitude)
                log_mag_max = np.max(OF_log_magnitude)
                img = (np.stack([OF_log_magnitude for _ in range(3)], axis=-1) - log_mag_min)
                if log_mag_max == log_mag_min:
                    img = np.zeros_like(img)
                else:
                    img = img/(log_mag_max - log_mag_min)

                img = img*255
                img = np.ndarray.astype(img, dtype=np.uint8)

                plt.imshow(img, interpolation='nearest', )

                if (GoalBee.INPUT_OF_ORIENTATION in OF) and (not args.ignorientation):
                    ss = 10
                    h, w = np.meshgrid(np.arange(OF_log_magnitude.shape[0]), np.arange(OF_log_magnitude.shape[1]))
                    OF_orientation = OF[GoalBee.INPUT_OF_ORIENTATION]
                    OF_magnitude = np.exp(OF_log_magnitude)
                    of_disp = np.transpose(OF_orientation*np.expand_dims(OF_magnitude, 0),
                                           axes=(0, 2, 1))
                    # inverted from image (height is top down) to np plot (y dim  bottom up)
                    plt.quiver(w[::ss, ::ss], h[::ss, ::ss],
                               of_disp[0, ::ss, ::ss],
                               -of_disp[1, ::ss, ::ss],
                               color='red',
                               scale=OF_scale[cam_name]*(max(w.shape)/ss),
                               )

                # plt.show()
                filename = os.path.join(img_dir, 'of_' + cam_name + '_' + str(t) + '.png')
                plt.savefig(filename, bbox_inches='tight')
                plt.close()
                of_img_files[cam_name].append(filename)
        for cam_name in of_img_files:
            files = of_img_files[cam_name]
            filename = os.path.join(output_dir, 'OF_gifed.gif')
            create_gif(image_paths=of_img_files[cam_name],
                       output_gif_path=filename,
                       duration=env.unwrapped.dt*1000*capture_interval,
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
