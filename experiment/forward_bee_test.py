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

    PARSER = argparse.ArgumentParser(
        description='run RL on envionrment'
    )

    # PARSER.add_argument("--env", action='store', required=False, default='HiBee-v0',
    #                    choices=('Beese-v0', 'HiBee-v0', 'ForwardBee-v0'),
    #                   help="RL gym class to run")
    PARSER.add_argument("--timesteps-per-epoch", type=int, required=False, default=1024,
                        help="number of timesteps to train for each epoch")
    PARSER.add_argument("--epochs", type=int, required=False, default=100,
                        help="number of epochs")
    PARSER.add_argument("--nsteps", type=int, required=False, default=512,
                        help="number of steps before learning step")
    PARSER.add_argument("--testjectories", type=int, required=False, default=100,
                        help="number of trajectories to collect each epoch")
    PARSER.add_argument("--dt", type=float, required=False, default=.1,
                        help="simulation timestep")

    args = PARSER.parse_args()
    DIR = os.path.dirname(os.path.dirname(__file__))
    output_dir = os.path.join(DIR, 'output', 'forw_bee_test')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    env = gym.make('ForwardBee-v0', dt=args.dt, )
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )

    model = MODEL('CnnPolicy', env, verbose=1, policy_kwargs=policy_kwargs,
                  # buffer_size=2048,  # for replay buffer methods
                  n_steps=args.nsteps,
                  )
    print(model.policy)
    for epoch in range(args.epochs):
        print('doing epoch', epoch)
        model.learn(total_timesteps=args.timesteps_per_epoch)
        print('finished training, getting trajectories')
        trajectories = []
        for _ in range(args.testjectories):
            poses = []
            obs, _ = env.reset(options={
                'initial_pos': ((-3., -2.), (-.5, .5), (-1., -1.5)) # tighter box
            })
            rwds = []
            done = False
            info = None
            while not done:
                action, _ = model.predict(observation=obs, deterministic=False)

                obs, rwd, done, term, info = env.step(action)
                rwds.append(rwd)
                poses.append(env.unwrapped.get_pose())

            print('ep length:', len(rwds))
            print('rwd mean:', sum(rwds)/len(rwds))

            trajectories.append({
                # 'poses': poses,
                'collided': info['collided'],
                'rewards': rwds,
                'positions': [np.array((pose.position.x_val, pose.position.y_val, pose.position.z_val))
                              for pose in poses],
                'orientations': [np.array((pose.orientation.x_val,
                                           pose.orientation.y_val,
                                           pose.orientation.z_val,
                                           pose.orientation.w_val,
                                           ))
                                 for pose in poses]
            })
        epoch_info = {
            'trajectories': trajectories
        }
        print('saving epoch', epoch, 'info')
        fname = os.path.join(output_dir, 'epoch_info' + str(epoch) + '.pkl')
        f = open(fname, 'wb')
        pkl.dump(epoch_info, f)
        f.close()
        print('saved epoch')

    print('saving all epochs')
    epoch_infos = []

    for epoch in range(args.epochs):
        fname = os.path.join(output_dir, 'epoch_info' + str(epoch) + '.pkl')
        f = open(fname, 'wb')
        epoch_infos.append(pkl.load(f))
        f.close()
    fname = os.path.join(output_dir, 'all_trajectories.pkl')
    f = open(fname, 'wb')
    pkl.dump(epoch_infos, f)
    f.close()
    print('saved epochs')
    while True:
        obs, _ = env.reset()
        rwds = []
        done = False
        while not done:
            action, _ = model.predict(observation=obs, deterministic=False)

            obs, rwd, done, term, info = env.step(action)
            rwds.append(rwd)
        print('ep length:', len(rwds))
        print('rwd mean:', sum(rwds)/len(rwds))
