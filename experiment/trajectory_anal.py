import os
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import argparse

DIR = os.path.dirname(os.path.dirname(__file__))

PARSER = argparse.ArgumentParser()
PARSER.add_argument('--traj-dir', action='store', required=False, default=os.path.join(DIR, 'output', 'forw_bee_test'),
                    help='place to look for traj files')
args = PARSER.parse_args()
traj_dir = args.traj_dir
print('analing trajectories from', traj_dir)
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
        fn = os.path.join(traj_dir, 'traj_' + str(i) + '.pkl')
        if not os.path.exists(fn):
            break
        f = open(fn, 'rb')
        epoch_infos.append(pkl.load(f))
        f.close()
        i += 1
medians = []
means = []
maxes = []
mins = []

rwd_medians = []
rwd_means = []
rwd_maxes = []
rwd_mins = []
xlim = (-5, 15)
# fig, ax = plt.subplots()
for epoch, trajs in enumerate(epoch_infos):
    for all_traj in True, False:
        # plot by swapping x and y
        fig, ax = plt.subplots()
        plt.xlabel('x axis')
        plt.ylabel('y axis')
        plt.gca().set_aspect('equal')
        ax.plot(xlim, [-2., -2.], color='black', linewidth=4)
        ax.plot(xlim, [2., 2.], color='black', linewidth=4)
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
            return traj[-1]['pose']['position'][0] - traj[0]['pose']['position'][0]


        trajs.sort(key=get_dist_traveled)

        if all_traj:
            plot_stuff = []
            for traj in trajs:
                kwargs = dict()
                last_info = traj[-1]['info']
                collided = last_info['collided']
                succ = last_info['succ']
                rewards = np.array([dic['reward'] for dic in traj])
                kwargs['color'] = 'red'
                kwargs['zorder'] = 1
                if not collided:
                    kwargs['color'] = 'yellow'
                    kwargs['zorder'] = 2
                if succ:
                    kwargs['color'] = 'green'
                    kwargs['zorder'] = 3
                plot_stuff.append((traj, kwargs))
        else:
            dists = np.array([get_dist_traveled(traj) for traj in trajs])
            medians.append(np.median(dists))
            maxes.append(np.max(dists))
            means.append(np.mean(dists))
            mins.append(np.min(dists))
            plot_stuff = (
                (trajs[len(trajs)//2], {'color': 'orange', 'label': 'median', }),
                (trajs[-1], {'color': 'green', 'label': 'best'})
            )
            rwds = np.array([sum(t['reward'] for t in traj) for traj in trajs])

            rwd_medians.append(np.median(rwds))
            rwd_maxes.append(np.max(rwds))
            rwd_means.append(np.mean(rwds))
            rwd_mins.append(np.min(rwds))
        for traj, kwargs in plot_stuff:
            positions = np.stack([traj[0]['old_pose']['position']] +
                                 [dic['pose']['position'] for dic in traj],
                                 axis=0)
            plt.plot(positions[:, 0], positions[:, 1], alpha=alpha, **kwargs)
        if not all_traj:
            plt.legend(loc='upper right', bbox_to_anchor=(.9, -.25))
        plt.title('epoch ' + str(epoch))
        plt.xlim(xlim)
        plt.savefig(os.path.join(plot_dir, ('all_' if all_traj else '') +
                                 'epoch_' + str(epoch) + '_trajectories.png'), bbox_inches='tight')
        plt.close()

plt.plot(means, color='blue', label='means')
plt.plot(medians, color='orange', label='median')
plt.plot(maxes, color='green', label='max')
plt.plot(mins, color='red', label='min')
plt.xlabel('epochs')
plt.ylabel('distance traveled')
plt.title("Distance traveled throughout training")

lower_bound = min(-1, plt.ylim()[0])
plt.ylim((lower_bound, plt.ylim()[1] + 1))
ylim = plt.ylim()
plt.legend()
plt.savefig(os.path.join(plot_dir, 'trajectory_summary.png'))
plt.close()

plt.plot(rwd_means, color='blue', label='means')
plt.plot(rwd_medians, color='orange', label='median')
plt.plot(rwd_maxes, color='green', label='max')
plt.plot(rwd_mins, color='red', label='min')
plt.xlabel('epochs')
plt.ylabel('reward sum')
plt.title("Rewards throughout training")
plt.legend()
plt.ylim(ylim)
plt.savefig(os.path.join(plot_dir, 'rwd_summary.png'))
plt.close()
