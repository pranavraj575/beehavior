import gymnasium as gym

from beehavior.networks.cnn import CNN


class CustomCNN(CNN):
    """
    https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, layers=2):
        channels = (32,) + tuple(64 for _ in range(layers))
        kernels = (8,) + tuple(3 for _ in range(layers))
        strides = (4,) + tuple(1 for _ in range(layers))
        paddings = (0,) + tuple(1 for _ in range(layers))
        super().__init__(observation_space=observation_space,
                         channels=channels,
                         kernels=kernels,
                         strides=strides,
                         paddings=paddings,
                         features_dim=features_dim,
                         )


if __name__ == '__main__':
    import argparse
    from stable_baselines3 import PPO as MODEL
    import beehavior

    PARSER = argparse.ArgumentParser(
        description='run RL on envionrment'
    )

    PARSER.add_argument("--env", action='store', required=False, default='HiBee-v0',
                        choices=('Beese-v0', 'HiBee-v0', 'ForwardBee-v0'),
                        help="RL gym class to run")
    PARSER.add_argument("--timesteps", type=int, required=False, default=10000,
                        help="number of timesteps to train for")
    args = PARSER.parse_args()

    env = gym.make(args.env, dt=.1, )
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )

    model = MODEL('CnnPolicy', env, verbose=1, policy_kwargs=policy_kwargs,
                  # buffer_size=2048,  # for replay buffer methods
                  n_steps=256,
                  )
    model.learn(total_timesteps=args.timesteps)
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
