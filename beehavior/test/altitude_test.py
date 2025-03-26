import gymnasium as gym

from beehavior.networks.cnn import CNN


class CustomCNN(CNN):
    """
    https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space=observation_space,
                         channels=(32, 64, 64),
                         kernels=(8, 4, 3),
                         strides=(4, 2, 1),
                         paddings=(0, 0, 1),
                         features_dim=features_dim,
                         )


if __name__ == '__main__':
    import beehavior
    from stable_baselines3 import PPO as MODEL

    env = gym.make('HiBee-v0', dt=.1, )

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )

    model = MODEL('CnnPolicy', env, verbose=1, policy_kwargs=policy_kwargs,
                  # buffer_size=1000,  # for replay buffer methods
                  n_steps=256,
                  )
    model.learn(total_timesteps=10000)
    for _ in range(100):
        obs, _ = env.reset()
        rwds = []
        done = False
        while not done:
            action, _ = model.predict(observation=obs, deterministic=False)
            obs, rwd, done, term, info = env.step(action)
            rwds.append(rwd)
        print('ep length:', len(rwds))
        print('rwd mean:', sum(rwds)/len(rwds))
