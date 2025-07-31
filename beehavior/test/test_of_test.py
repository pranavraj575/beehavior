"""
testing test enviornmet
"""
if __name__ == '__main__':
    import beehavior
    import gymnasium as gym
    from stable_baselines3 import A2C as MODEL

    env = gym.make('Test-v0')
    model = MODEL('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(observation=obs)
        print('TARGET DIRECTION:', obs, '; ACTION:', action)
        obs, rwd, done, term, info = env.step(action)

