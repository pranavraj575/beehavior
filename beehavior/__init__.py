import gymnasium as gym

gym.envs.register(
    id='Test-v0',
    entry_point='beehavior.envs.test:Test',
    max_episode_steps=10000,
)
