import os
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

DIR = os.path.dirname(os.path.dirname(__file__))
traj_dir = os.path.join(DIR, 'output', 'forw_bee_test')
plot_dir = os.path.join(traj_dir, 'plots')
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

if os.path.exists(os.path.join(traj_dir, 'all_trajectories.pkl')):
    f = open(os.path.join(traj_dir, 'all_trajectories.pkl'), 'rb')
    epoch_infos = pkl.load(f)
    f.close()
else:
    epoch_infos = []
    i = 0
    while True:
        fn = os.path.join(traj_dir, 'epoch_info' + str(i) + '.pkl')
        if not os.path.exists(fn):
            break
        f = open(fn, 'rb')
        epoch_infos.append(pkl.load(f))
        f.close()
        i += 1
medians = []
means = []
maxes = []
# fig, ax = plt.subplots()
for epoch, epoch_info in enumerate(epoch_infos):
    trajs = epoch_info['trajectories']
    # plot by swapping x and y
    fig, ax = plt.subplots()
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.gca().set_aspect('equal')
    ax.plot([-5, 15], [-2., -2.], color='black', linewidth=4)
    ax.plot([-5, 15], [2., 2.], color='black', linewidth=4)
    for c, r, alpha in (
            ((-92.30, 1.04), .5, 1),
            ((-92.30, 1.04), 1, .3),
            ((-81.40, -.66), .5, 1),
            ((-81.40, -.66), 1, .3),
    ):
        c = c[0] + 94., -c[1]
        ax.add_patch(plt.Circle(c, r, alpha=alpha))
    alpha = 1


    def get_dist_traveled(traj):
        return traj['positions'][-1][0] - traj['positions'][0][0]


    trajs.sort(key=get_dist_traveled)
    medians.append(get_dist_traveled(trajs[len(trajs)//2]))
    maxes.append(get_dist_traveled(trajs[-1]))
    means.append(sum(get_dist_traveled(traj) for traj in trajs)/len(trajs))
    for traj, kwargs in (  # (trajs[0], {'color': 'red', 'label': 'worst'}),
            (trajs[len(trajs)//2], {'color': 'orange', 'label': 'median', }),
            (trajs[-1], {'color': 'green', 'label': 'best'})):
        collided = traj['collided']
        rewards = np.array(traj['rewards'])

        positions = np.stack(traj['positions'], axis=0)

        plt.plot(positions[:, 0], positions[:, 1], alpha=alpha, **kwargs)
    plt.legend(loc='upper right', bbox_to_anchor=(.9, -.25))
    plt.title('epoch ' + str(epoch))
    plt.savefig(os.path.join(plot_dir, 'epoch_' + str(epoch) + '_trajectories.png'), bbox_inches='tight')
    plt.close()

plt.plot(means, color='blue', label='means')
plt.plot(medians, color='orange', label='median')
plt.plot(maxes, color='green', label='max')
plt.xlabel('epochs')
plt.ylabel('distance traveled')
plt.legend()
plt.savefig(os.path.join(plot_dir, 'trajectory_summary.png'))
plt.close()
