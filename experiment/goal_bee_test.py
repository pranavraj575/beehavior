import ast

import gymnasium as gym
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
    from experiment.utils import load_model, get_model_history_srt, clear_model_history

    DIR = os.path.dirname(os.path.dirname(__file__))

    PARSER = argparse.ArgumentParser(
        description='run RL on envionrment'
    )

    # PARSER.add_argument("--env", action='store', required=False, default='HiBee-v0',
    #                    choices=('Beese-v0', 'HiBee-v0', 'ForwardBee-v0'),
    #                   help="RL gym class to run")

    PARSER.add_argument("--ident", action='store', required=False, default='goal_bee_test',
                        help="test identification")
    PARSER.add_argument("--timesteps-per-epoch", type=int, required=False, default=1024,
                        help="number of timesteps to train for each epoch")
    PARSER.add_argument("--epochs", type=int, required=False, default=50,
                        help="number of epochs")
    PARSER.add_argument("--ckpt-freq", type=int, required=False, default=1,
                        help="frequency to save model")
    PARSER.add_argument("--test-freq", type=int, required=False, default=None,
                        help="frequency to check trajectories, defaults to same as --ckpt-freq")
    PARSER.add_argument("--nsteps", type=int, required=False, default=1024,
                        help="number of steps before learning step")
    PARSER.add_argument("--testjectories", type=int, required=False, default=32,
                        help="number of trajectories to collect each epoch")
    PARSER.add_argument("--dt", type=float, required=False, default=.05,
                        help="simulation timestep")
    PARSER.add_argument("--history-steps", type=int, required=False, default=1,
                        help="steps to see in history")

    PARSER.add_argument('--action-type', action='store', required=False, default=GoalBee.ACTION_ACCELERATION_XY,
                        choices=(GoalBee.ACTION_VELOCITY,
                                 GoalBee.ACTION_VELOCITY_XY,
                                 GoalBee.ACTION_ACCELERATION,
                                 GoalBee.ACTION_ACCELERATION_XY,
                                 GoalBee.ACTION_ROLL_PITCH_THRUST,
                                 GoalBee.ACTION_ROTORS,
                                 ),
                        help='action space to use: velocity, acceleration, or rpt')
    PARSER.add_argument("--include-raw-of", action='store_true', required=False,
                        help="include raw OF in input")
    PARSER.add_argument("--include-log-of", action='store_true', required=False,
                        help="include log OF in input")
    PARSER.add_argument("--include-of-orientation", action='store_true', required=False,
                        help="include OF orientation in input")
    PARSER.add_argument("--include-of-vector", action='store_true', required=False,
                        help="include OF vector in input")
    PARSER.add_argument("--include-depth", action='store_true', required=False,
                        help="include depth in input")

    PARSER.add_argument("--network", action='store', required=False,
                        default=os.path.join(DIR, 'beehavior', 'networks', 'configs', 'shrimple_gc.txt'),
                        help="network config file to use (look at beehavior/networks/nn_from_config.py)")
    PARSER.add_argument("--pol-val-net", type=int, nargs='*', required=False, default=[64, ],
                        help="hidden layer list of policy and value nets")

    PARSER.add_argument("--recollect", action='store_true', required=False,
                        help="go through all saved models after training and collect <--testjectories> trajectories")
    PARSER.add_argument("--save-model-history", type=int, required=False, default=-1,
                        help="number of past models to save (-1 for all)")
    PARSER.add_argument("--testing-tunnel", type=int, nargs='+', required=False, default=[1],
                        help="index of tunnels to test in, 1 is the normal one")
    PARSER.add_argument("--nondeterministic-testjectory", action='store_true', required=False,
                        help='use randomness in testjectory')

    PARSER.add_argument("--cameras", nargs='+', type=str, required=False, default=['front', 'bottom'],
                        help='cameras to use (if not all)', choices=('front', 'bottom'))

    PARSER.add_argument("--goals", nargs='+', type=str, required=False, default=[GoalBee.GOAL_FORWARD],
                        help='cameras to use (if not all)',
                        choices=(GoalBee.GOAL_FORWARD,
                                 GoalBee.GOAL_HOVER,
                                 GoalBee.GOAL_STATION_KEEP,
                                 GoalBee.GOAL_LAND_ON,
                                 )
                        )

    PARSER.add_argument("--dictionary-space", action='store_true', required=False,
                        help='gym environment is a dictionary, this should make model faster, but cannot use this for SHAP')

    PARSER.add_argument("--reset", action='store_true', required=False,
                        help="reset training")
    PARSER.add_argument("--display", type=int, required=False, default=None,
                        help="skip training and run specified saved model (-1 for most recent) on all <--testing-tunnel>s")
    args = PARSER.parse_args()
    concat_obs = not args.dictionary_space

    test_freq = args.ckpt_freq if args.test_freq is None else args.test_freq
    network_file = args.network

    testing_tunnels = sorted(set(args.testing_tunnel))
    of_cameras = tuple(sorted(set(args.cameras)))
    init_goals = {(g,): 1/len(set(args.goals)) for g in set(args.goals)}

    img_input_space = []
    if args.include_log_of:
        img_input_space.append(GoalBee.INPUT_LOG_OF)
    if args.include_of_orientation:
        img_input_space.append(GoalBee.INPUT_OF_ORIENTATION)
    if args.include_raw_of:
        img_input_space.append(GoalBee.INPUT_RAW_OF)
    if args.include_of_vector:
        img_input_space.append(GoalBee.INPUT_OF_VECTOR)
    if args.include_depth:
        img_input_space.append(GoalBee.INPUT_INV_DEPTH_IMG)
    if not img_input_space:
        raise Exception('need to add at least one image input')

    ident = args.ident
    if concat_obs:
        ident += '_cat_obs'
    ident += '_in_'
    for key in (GoalBee.INPUT_RAW_OF,
                GoalBee.INPUT_LOG_OF,
                GoalBee.INPUT_OF_ORIENTATION,
                GoalBee.INPUT_OF_VECTOR,
                ):
        if key in img_input_space:
            ident += 'y'
        else:
            ident += 'n'
    if GoalBee.INPUT_INV_DEPTH_IMG in img_input_space:
        ident += 'd'
    if set(of_cameras) != {'bottom', 'front'}:
        ident += '_cams_' + '_'.join(of_cameras)
    ident += '_gls_' + '_'.join(sorted(set(args.goals)))
    ident += '_act_' + args.action_type
    ident += '_k_' + str(args.history_steps)
    ident += '_dt_' + str(args.dt).replace('.', '_')
    ident += '_epoch_stp_' + str(args.timesteps_per_epoch)
    ident += '_nstep_' + str(args.nsteps)
    ident += '_net_' + os.path.basename(network_file)[:os.path.basename(network_file).find('.')]
    ident += '_pol_val_' + '_'.join([str(h) for h in args.pol_val_net])

    output_dir: str = os.path.join(DIR, 'output', ident)
    traj_dir = os.path.join(output_dir, 'trajectories')
    model_dir = os.path.join(output_dir, 'past_models')
    for d in (output_dir, traj_dir, model_dir):
        if not os.path.exists(d): os.makedirs(d)
    print('saving to', output_dir)
    env_config = {'name': 'GoalBee-v0',
                  'kwargs': dict(
                      dt=args.dt,
                      input_img_space=img_input_space,
                      action_type=args.action_type,
                      img_history_steps=args.history_steps,
                      concatenate_observations=concat_obs,
                      of_cameras=of_cameras,
                      initial_goals=init_goals,
                  )}
    f = open(os.path.join(output_dir, 'env_config.txt'), 'w')
    f.write(str(env_config))
    f.close()
    env = gym.make(env_config['name'],
                   **env_config['kwargs'],
                   )
    f = open(network_file, 'r')
    structure = ast.literal_eval(f.read())
    policy_kwargs = dict(
        features_extractor_class=CustomNN,
        features_extractor_kwargs=dict(structure=structure,
                                       ksp=env.unwrapped.get_ksp() if concat_obs else None,
                                       ),
        net_arch=dict(pi=args.pol_val_net, vf=args.pol_val_net),
    )
    f.close()

    model = MODEL('MultiInputPolicy',
                  env,
                  verbose=1,
                  policy_kwargs=policy_kwargs,
                  n_steps=args.nsteps,
                  )
    epoch_init = 0
    if not args.reset:
        past_models = get_model_history_srt(model_dir=model_dir)
        if past_models:
            model_file, info_file = past_models[-1]
            model, info = load_model(MODEL=MODEL, model_file=model_file, info_file=info_file, env=env)
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
        ((-4., -3.), (-4.8, -5.8), (-2., -6.9)),  # converging/diverging tunnel
        ((-4., -3.), (-.5, .5), (-2., -6.9)),  # normal tunnel
        ((-4., -3.), (5.5, 6.5), (-2., -6.9)),
        ((-4., -3.), (10.7, 11.3), (-2., -6.9)),  # narrow tunnel
        ((-4., -3.), (14.7, 15.3), (-2., -6.9)),
        ((-4., -3.), (19, 25), (-2., -6.9)),
        ((-4., -3.), (30, 33), (-2., -6.9)),
        ((-4., -3.), (38.5, 39.5), (-2., -6.9))  # empty tunnel
    ]


    def collect_testjectory(model, env, tunnel_idx):
        steps = []
        obs, info = env.reset(options={
            'initial_pos': initial_positions[tunnel_idx]
        })
        old_pose = env.unwrapped.get_pose()
        done = False
        while not done:
            action, _ = model.predict(observation=obs, deterministic=not args.nondeterministic_testjectory)

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
        return steps


    def collect_testjectories(model, env, testjectories=None, num_to_collect=None, debug=False):
        if testjectories is None:
            testjectories = dict()
        if num_to_collect is None:
            num_to_collect = args.testjectories
        for tunnel_idx in testing_tunnels:
            trajectories = testjectories.get(tunnel_idx, [])
            while len(trajectories) < num_to_collect:
                traj = collect_testjectory(model=model, env=env, tunnel_idx=tunnel_idx)
                trajectories.append(traj)
                if debug:
                    rwds = [dic['reward'] for dic in traj]
                    print('ep length:', len(rwds), ';',
                          'rwd sum:', sum(rwds), ';',
                          traj[-1]['info'],
                          )
            testjectories[tunnel_idx] = trajectories
        return testjectories


    for epoch in range(epoch_init, args.epochs):
        if args.display is not None:
            # skip training
            break
        traj_filename = os.path.join(traj_dir, 'traj_' + str(epoch) + '.pkl')
        model_filename = os.path.join(model_dir, 'model_epoch_' + str(epoch) + '.pkl')
        info_filename = os.path.join(model_dir, 'info_epoch_' + str(epoch) + '.pkl')

        print('doing epoch', epoch)
        model.learn(total_timesteps=args.timesteps_per_epoch,
                    reset_num_timesteps=False,
                    progress_bar=True)
        print('finished training')

        if (not (epoch + 1)%args.ckpt_freq):
            print('saving model and training info in', model_dir)
            model.save(model_filename)

            f = open(info_filename, 'wb')
            pkl.dump({'epochs_trained': epoch + 1}, f)
            f.close()
            clear_model_history(save_model_history=args.save_model_history, model_dir=model_dir)
        if args.testjectories and (not (epoch + 1)%test_freq):
            print('getting trajectories for epoch', epoch)
            testjectories = collect_testjectories(model=model,
                                                  env=env,
                                                  testjectories=None,
                                                  num_to_collect=args.testjectories,
                                                  debug=True,
                                                  )
            print('saving trajectories of epoch', epoch, 'info to ', traj_filename)
            f = open(traj_filename, 'wb')
            pkl.dump(testjectories, f)
            f.close()
            print('saved trajectories')

    if args.recollect and args.testjectories and (args.display is None):
        for epoch in range(args.epochs):
            if (not (epoch + 1)%test_freq):
                traj_filename = os.path.join(traj_dir, 'traj_' + str(epoch) + '.pkl')
                model_filename = os.path.join(model_dir, 'model_epoch_' + str(epoch) + '.pkl')
                info_filename = os.path.join(model_dir, 'info_epoch_' + str(epoch) + '.pkl')
                if os.path.exists(model_filename):
                    model, info = load_model(MODEL=MODEL, model_file=model_filename, info_file=info_filename, env=env)
                    print('collecting trajectories for epoch', epoch)
                else:
                    print('model for epoch', epoch, ':', model_filename, 'not saved')
                    continue
                testjectories = None
                if os.path.exists(traj_filename):
                    f = open(traj_filename, 'rb')
                    testjectories = pkl.load(f)
                    f.close()
                    print('loaded previous trajectories')
                testjectories = collect_testjectories(model=model,
                                                      env=env,
                                                      testjectories=testjectories,
                                                      num_to_collect=args.testjectories,
                                                      debug=True,
                                                      )

                print('saving trajectories of epoch', epoch, 'info to ', traj_filename)
                f = open(traj_filename, 'wb')
                pkl.dump(testjectories, f)
                f.close()
                print('saved trajectories')
    if args.display is not None:
        if args.display < 0:
            past_models = get_model_history_srt(model_dir=model_dir)
            if past_models:
                model_filename, info_filename = past_models[-1]
            else:
                raise Exception('no models saved')
        else:
            model_filename = os.path.join(model_dir, 'model_epoch_' + str(args.display) + '.pkl')
            info_filename = os.path.join(model_dir, 'info_epoch_' + str(args.display) + '.pkl')

        if os.path.exists(model_filename):
            model, info = load_model(MODEL=MODEL, model_file=model_filename, info_file=info_filename, env=env)
            print('loading model', model_filename)
        else:
            raise Exception('model', model_filename, 'not saved')

        testjectories = collect_testjectories(model=model,
                                              env=env,
                                              testjectories=None,
                                              num_to_collect=1,
                                              debug=True,
                                              )
