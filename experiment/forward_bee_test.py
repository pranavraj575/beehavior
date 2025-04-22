import gymnasium as gym

from beehavior.networks.cnn import CNN


class CustomCNN(CNN):
    """
    https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, ffn_hidden=(), features_dim: int = 128):
        channels = (64, 128, 128, 64, 32)
        kernels = (9, 5, 3, 3, 3)
        strides = (3, 1, 1, 1, 1)
        paddings = (0, 2, 1, 1, 1)
        maxpools = (True, True, False, False, True)
        super().__init__(observation_space=observation_space,
                         channels=channels,
                         kernels=kernels,
                         strides=strides,
                         paddings=paddings,
                         ffn_hidden_layers=ffn_hidden,
                         features_dim=features_dim,
                         maxpools=maxpools,
                         )


if __name__ == '__main__':
    import argparse
    import numpy as np
    from stable_baselines3 import PPO as MODEL
    import os
    import pickle as pkl
    import beehavior
    from beehavior.envs.forward_bee import ForwardBee

    PARSER = argparse.ArgumentParser(
        description='run RL on envionrment'
    )

    # PARSER.add_argument("--env", action='store', required=False, default='HiBee-v0',
    #                    choices=('Beese-v0', 'HiBee-v0', 'ForwardBee-v0'),
    #                   help="RL gym class to run")

    PARSER.add_argument("--ident", action='store', required=False, default='forw_bee_test',
                        help="test identification")
    PARSER.add_argument("--timesteps-per-epoch", type=int, required=False, default=512,
                        help="number of timesteps to train for each epoch")
    PARSER.add_argument("--epochs", type=int, required=False, default=100,
                        help="number of epochs")
    PARSER.add_argument("--nsteps", type=int, required=False, default=512,
                        help="number of steps before learning step")
    PARSER.add_argument("--testjectories", type=int, required=False, default=32,
                        help="number of trajectories to collect each epoch")
    PARSER.add_argument("--dt", type=float, required=False, default=.1,
                        help="simulation timestep")
    PARSER.add_argument("--history-steps", type=int, required=False, default=2,
                        help="steps to see in history")
    PARSER.add_argument('--action-type', action='store', required=False, default=ForwardBee.ACTION_VELOCITY,
                        choices=(ForwardBee.ACTION_VELOCITY,
                                 ForwardBee.ACTION_VELOCITY_XY,
                                 ForwardBee.ACTION_ACCELERATION,
                                 ForwardBee.ACTION_ROLL_PITCH_YAW,
                                 ),
                        help='action space to use: velocity, acceleration, or rpy')
    PARSER.add_argument("--include-raw-of", action='store_true', required=False,
                        help="include raw OF in input")
    PARSER.add_argument("--include-log-of", action='store_true', required=False,
                        help="include log OF in input")
    PARSER.add_argument("--include-of-orientation", action='store_true', required=False,
                        help="include OF orientation in input")
    PARSER.add_argument("--include-depth", action='store_true', required=False,
                        help="include depth in input")
    PARSER.add_argument("--save-model-history", type=int, required=False, default=1,
                        help="number of past models to save (-1 for all)")
    PARSER.add_argument("--testing-tunnel", type=int, nargs='+', required=False, default=[1],
                        help="index of tunnels to test in, 1 is the normal one")
    PARSER.add_argument("--reset", action='store_true', required=False,
                        help="reset training")
    args = PARSER.parse_args()
    testing_tunnels = sorted(set(args.testing_tunnel))

    img_input_space = []
    if args.include_log_of:
        img_input_space.append(ForwardBee.INPUT_LOG_OF)
    if args.include_of_orientation:
        img_input_space.append(ForwardBee.INPUT_OF_ORIENTATION)
    if args.include_raw_of:
        img_input_space.append(ForwardBee.INPUT_RAW_OF)
    if args.include_depth:
        img_input_space.append(ForwardBee.INPUT_INV_DEPTH_IMG)
    if not img_input_space:
        raise Exception('need to add at least one image input')
    ident = args.ident
    ident += '_in_'
    for key in (ForwardBee.INPUT_RAW_OF,
                ForwardBee.INPUT_LOG_OF,
                ForwardBee.INPUT_OF_ORIENTATION,
                ForwardBee.INPUT_INV_DEPTH_IMG,
                ):
        if key in img_input_space:
            ident += 'y'
        else:
            ident += 'n'
    ident += '_act_' + args.action_type
    ident += '_k_' + str(args.history_steps)
    ident += '_dt_' + str(args.dt).replace('.', '_')
    ident += '_tst_' + '_'.join([str(t) for t in testing_tunnels])

    DIR = os.path.dirname(os.path.dirname(__file__))
    output_dir: str = os.path.join(DIR, 'output', ident)
    traj_dir = os.path.join(output_dir, 'trajectories')
    model_dir = os.path.join(output_dir, 'past_models')
    for d in (output_dir, traj_dir, model_dir):
        if not os.path.exists(d): os.makedirs(d)
    print('saving to', output_dir)
    env = gym.make('ForwardBee-v0',
                   dt=args.dt,
                   input_img_space=img_input_space,
                   action_type=args.action_type,
                   img_history_steps=args.history_steps,
                   )
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )


    def epoch_from_name(basename) -> int:
        if basename.startswith('model_'):
            return int(basename[12:-4])  # model name is model_epoch_<number>.pkl
        elif basename.startswith('info_'):
            return int(basename[11:-4])  # info name is info_epoch_<number>.pkl
        else:
            raise Exception('does not match known model or info files:', basename)


    def get_model_history_srt():
        filenames = os.listdir(model_dir)
        if not filenames:
            return ()
        largest = max(map(epoch_from_name, filenames))
        stuff = [[None, None] for _ in range(largest + 1)]
        for fn in filenames:
            idx = epoch_from_name(fn)
            if fn.startswith('model_'):
                stuff[idx][0] = os.path.join(model_dir, fn)
            elif fn.startswith('info_'):
                stuff[idx][1] = os.path.join(model_dir, fn)
        return tuple(tuple(s) for s in stuff if s[0] is not None)


    def clear_model_history():
        past_models = get_model_history_srt()
        if args.save_model_history > -1 and len(past_models) > args.save_model_history:
            for model_name, info_name in past_models[:len(past_models) - args.save_model_history]:
                os.remove(model_name)
                os.remove(info_name)


    model = MODEL('CnnPolicy', env, verbose=1, policy_kwargs=policy_kwargs,
                  # buffer_size=2048,  # for replay buffer methods
                  n_steps=args.nsteps,
                  )
    epoch_init = 0
    if not args.reset:
        # TODO: LOADING AND SAVING DO NOT WORK
        past_models = get_model_history_srt()
        if past_models:
            model_file, info_file = past_models[-1]
            model = MODEL.load(model_file, env=env)
            f = open(info_file, 'rb')
            info = pkl.load(f)
            f.close()
            epoch_init = info['epochs_trained']

            print('loading from', model_file)
            print('epochs already trained:', epoch_init)

    print(model.policy)


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


    initial_positions = [
        ((-3., -2.), (-4.8, -5.8), (-1., -1.5)),  # converging/diverging tunnel
        ((-3., -2.), (-.5, .5), (-1., -1.5)),  # normal tunnel
        ((-3., -2.), (5.5, 6.5), (-1., -1.5)),
        ((-3., -2.), (10.7, 11.3), (-1., -1.5)),  # narrow tunnel
        ((-3., -2.), (14.7, 15.3), (-1., -1.5)),
        ((-3., -2.), (19, 25), (-1., -1.5)),
        ((-3., -2.), (30, 33), (-1., -1.5)),
        ((-3., -2.), (38.5, 39.5), (-1., -1.5))  # empty tunnel
    ]
    for epoch in range(epoch_init, args.epochs):
        print('doing epoch', epoch)
        model.learn(total_timesteps=args.timesteps_per_epoch,
                    reset_num_timesteps=False,
                    progress_bar=True)
        print('finished training, getting trajectories')
        trajectories = []
        for _ in range(args.testjectories):
            for tunnel_idx in testing_tunnels:
                steps = []
                obs, info = env.reset(options={
                    'initial_pos': initial_positions[tunnel_idx]
                })
                old_pose = env.unwrapped.get_pose()
                done = False
                while not done:
                    action, _ = model.predict(observation=obs, deterministic=False)

                    obs, rwd, done, term, info = env.step(action)
                    pose = env.unwrapped.get_pose()
                    steps.append({
                        'old_pose': pose_to_dic(old_pose),
                        'action': action,
                        'reward': rwd,
                        'pose': pose_to_dic(pose),
                        'info': info,
                    })
                    old_pose = pose
                rwds = [dic['reward'] for dic in steps]
                print('ep length:', len(rwds))
                print('rwd sum:', sum(rwds))

                trajectories.append(steps)
        print('saving trajectories of epoch', epoch, 'info')
        fname = os.path.join(traj_dir, 'traj_' + str(epoch) + '.pkl')
        f = open(fname, 'wb')
        pkl.dump(trajectories, f)
        f.close()
        print('saved trajectory')
        print('saving model and training info')
        model.save(os.path.join(model_dir, 'model_epoch_' + str(epoch) + '.pkl'))
        fname = os.path.join(model_dir, 'info_epoch_' + str(epoch) + '.pkl')
        f = open(fname, 'wb')
        pkl.dump({'epochs_trained': epoch + 1}, f)
        f.close()
        clear_model_history()
        print('saved model and training info')

    while False:
        obs, _ = env.reset()
        rwds = []
        done = False
        while not done:
            action, _ = model.predict(observation=obs, deterministic=False)

            obs, rwd, done, term, info = env.step(action)
            rwds.append(rwd)
        print('ep length:', len(rwds))
        print('rwd mean:', sum(rwds)/len(rwds))
