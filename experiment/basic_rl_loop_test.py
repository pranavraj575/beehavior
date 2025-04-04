import gymnasium as gym

from beehavior.networks.cnn import CNN

if __name__ == '__main__':
    import gymnasium as gym

    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    import matplotlib.pyplot as plt

    # Parallel environments
    # vec_env = make_vec_env("CartPole-v1")
    env = gym.make('CartPole-v1')
    steps_per_epoch = 512

    for iters_per_epoch in 1, 2, 4, 8:
        model = PPO("MlpPolicy", env, verbose=0, n_steps=steps_per_epoch//iters_per_epoch)
        record = []
        for epoch in range(50):
            model.learn(total_timesteps=steps_per_epoch, reset_num_timesteps=False, progress_bar=False)
            stuff = []
            for tests in range(10):
                obs, info = env.reset()  # vec_env.reset()
                term = False
                rwd_sum = 0
                while not term:
                    action, _states = model.predict(obs)
                    obs, rewards, term, _, info = env.step(action)
                    rwd_sum += rewards
                stuff.append(rwd_sum)
            avg_rwd = sum(stuff)/len(stuff)
            print('epoch', epoch)
            print('rwd:', avg_rwd)
            record.append((model.num_timesteps, avg_rwd))
        plt.plot([r[0] for r in record], [r[1] for r in record], label=str(iters_per_epoch) + ' iter/epoch')
    plt.legend()
    plt.xlabel('timesteps')
    plt.ylabel('episode rewards')
    plt.show()
