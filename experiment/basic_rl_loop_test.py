import gymnasium as gym

from beehavior.networks.cnn import CNN

if __name__ == '__main__':
    import gymnasium as gym
    import os

    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    import matplotlib.pyplot as plt

    # Parallel environments
    # vec_env = make_vec_env("CartPole-v1")
    env = gym.make('CartPole-v1')
    steps_per_epoch = 256


    def test(model, env, n=10):
        stuff = []
        for tests in range(n):
            obs, info = env.reset()  # vec_env.reset()
            term = False
            rwd_sum = 0
            while not term:
                action, _states = model.predict(obs)
                obs, rewards, term, _, info = env.step(action)
                rwd_sum += rewards
            stuff.append(rwd_sum)
        return sum(stuff)/len(stuff)


    # save test
    DIR = os.path.dirname(os.path.dirname(__file__))
    save_file = os.path.join(DIR, 'output', 'model.pkl')
    rwds = []
    model = PPO("MlpPolicy", env, verbose=0, n_steps=steps_per_epoch)
    for i in range(5):
        if i > 0:
            # this DOES work
            model = PPO.load(save_file, env=env)
            # this does NOT work
            # model = PPO("MlpPolicy", env, verbose=0, n_steps=steps_per_epoch);model.load(save_file)
        for epoch in range(5):
            model.learn(total_timesteps=steps_per_epoch, reset_num_timesteps=False, progress_bar=False)
            avg_rwd = test(model, env, 10)
            rwds.append(avg_rwd)
            print(epoch)
        model.save(save_file)
    os.remove(save_file)
    plt.plot(rwds)
    plt.xlabel('epochs')
    plt.ylabel('avg rewards')
    plt.show()

    # iter per epoch test

    for iters_per_epoch in 1, 2, 4, 8:
        model = PPO("MlpPolicy", env, verbose=0, n_steps=steps_per_epoch//iters_per_epoch)
        record = []
        for epoch in range(50):
            model.learn(total_timesteps=steps_per_epoch, reset_num_timesteps=False, progress_bar=False)
            avg_rwd = test(model, env, 10)
            print('epoch', epoch)
            print('rwd:', avg_rwd)
            record.append((model.num_timesteps, avg_rwd))
        plt.plot([r[0] for r in record], [r[1] for r in record], label=str(iters_per_epoch) + ' iter/epoch')
    plt.legend()
    plt.xlabel('timesteps')
    plt.ylabel('episode rewards')
    plt.show()
