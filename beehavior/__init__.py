import gymnasium as gym

gym.envs.register(
    id='Test-v0',
    entry_point='beehavior.envs.test:Test',
    max_episode_steps=10000,
)
gym.envs.register(
    id='Beese-v0',
    entry_point='beehavior.envs.beese_class:BeeseClass',
    max_episode_steps=10000,
)

gym.envs.register(
    id='HiBee-v0',
    entry_point='beehavior.envs.hi_bee:HiBee',
    max_episode_steps=10000,
)