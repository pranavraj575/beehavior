import os, shutil
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import argparse
from PIL import Image


def create_gif(image_paths, output_gif_path, duration=200):
    images = [Image.open(image_path) for image_path in image_paths]
    # Save as GIF
    images[0].save(
        output_gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,  # 0 means infinite loop
    )


DIR = os.path.dirname(os.path.dirname(__file__))

PARSER = argparse.ArgumentParser()
PARSER.add_argument('--load-dir', action='store', required=True,
                    help='place that model saved')
PARSER.add_argument('--keep-individuals', action='store_true', required=False,
                    help='Dont delete individual traj files')
PARSER.add_argument('--avg-vel', action='store_true', required=False,
                    help='calculate avg vel')
PARSER.add_argument('--traj-freq', type=int, required=False, default=1,
                    help='frequency to look for traj files')
PARSER.add_argument('--dx', type=float, required=False, default=.5,
                    help='dx to use for space averaging positions/velocities')

args = PARSER.parse_args()
dx = args.dx
dt = .1
load_dir = args.load_dir
traj_dir = os.path.join(load_dir, 'trajectories')
print('analing trajectories from', traj_dir)
plot_dir = os.path.join(load_dir, 'plots')
print('plotting to', plot_dir)

individual_traj_dir = os.path.join(plot_dir, 'individual_trajectories_obtained_from_testing')
if not os.path.exists(individual_traj_dir):
    os.makedirs(individual_traj_dir)
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

epoch_infos = []
i = 0
last_seen = 0
while True:
    fn = os.path.join(traj_dir, 'traj_' + str(i) + '.pkl')
    if i - last_seen > args.traj_freq*10 and (not os.path.exists(fn)):
        break
    if os.path.exists(fn):
        f = open(fn, 'rb')
        epoch_infos.append(pkl.load(f))
        f.close()
        last_seen = i
    else:
        epoch_infos.append(dict())
    i += 1
if not epoch_infos:
    raise Exception('empty directory:', traj_dir)

# set of tunnel infos, specifically walls and obstacles
all_test_tunnel_infos = [
    # squished tunnel
    {
        'walls': [[(-5, -8.35), (3.85, -8.35), (9.35, -6.9), (15.05, -8.35), (20.85, -6.9)],
                  [(-5, -3.15), (3.85, -3.15), (9.35, -4.6), (15.05, -3.15), (20.85, -4.6)]],
        'xlim': (-3, 21),
        'obs': [],
    },
    # basic default tunnel with two obstacles
    {
        # list of vertex lists
        'walls': [[(-5, -2.1), (25, -2.1)], [(-5, 2.1), (25, 2.1)]],
        # list of (center, width, height (for ellipse, x axis, y axis))
        'obs': [((-9230, -22250), (1, 1)),
                ((-8140, -22080), (1, 1))
                ],
    },
    {
        # list of vertex lists
        'walls': [[(-5, 3.14), (25, 3.14)], [(-5, 8.8), (25, 8.8)], ],
        # list of (center, width, height (for ellipse, x axis, y axis))
        'obs': [
            ((-9230, -21530), (1, 1)),
            ((-8030, -21300), (1, 1)),
            ((-7900, -21790), (1, 1)),
        ],
    },
    None,
    None,
    None,
    None,
    # empty tunnel
    {
        'walls': [[(-5, 36.52), (25, 36.52)], [(-5, 41.39), (25, 41.39)]],
        'obs': [],
    }
]

common_info = {
    'xlim': (-5, 21),
    'finish_x': 20,
}

for tunnel_idx, test_tunnel_info in enumerate(all_test_tunnel_infos):
    if test_tunnel_info is None:
        continue

    # filter out the trajectories that are tested in this tunnel
    relevant_epoch_infos = [stuff.get(tunnel_idx, []) for stuff in epoch_infos]
    if not any(relevant_epoch_infos):
        continue
    epochs = []

    prop_successful = []

    medians = []
    means = []
    maxes = []
    mins = []

    rwd_medians = []
    rwd_means = []
    rwd_maxes = []
    rwd_mins = []

    test_tunnel_info.update(common_info)
    walls = test_tunnel_info['walls']
    xlim = test_tunnel_info['xlim']

    initial_wall_y = [pt_list[0][1] for pt_list in walls]
    initial_wall_y = min(initial_wall_y), max(initial_wall_y)


    # fig, ax = plt.subplots()

    def plt_env():
        fig, ax = plt.subplots()
        for pt_list in walls:
            ax.plot([wx for (wx, wy) in pt_list],
                    [wy for (wx, wy) in pt_list],
                    color='black', linewidth=4)
        ylim = plt.ylim()  # y limits should be bounded by walls
        for effective_bnd in (False, True):
            alpha = [1, .5][int(effective_bnd)]
            for c, (w, h) in test_tunnel_info['obs']:
                if effective_bnd:
                    # drone has a radius of about 1m, so obstacles need to be elongated by a radius of .5,
                    #   representing a crash if drone COM is within this area
                    # thus, increase width and height (diameter) by 1 each

                    w += 1
                    h += 1
                c = (c[0] + 9390)/100, (c[1] + 22145)/100
                # ax.add_patch(plt.Circle(c, r, alpha=alpha))
                ax.add_patch(Ellipse(xy=c,
                                     width=w,
                                     height=h,
                                     angle=0,
                                     alpha=alpha,
                                     ))

        xbnd = test_tunnel_info['finish_x']
        ax.plot([xbnd, xbnd], plt.ylim(),
                color='black', linewidth=2, linestyle='--'
                )
        return ylim


    fnames = []
    for epoch, trajs in enumerate(relevant_epoch_infos):
        xs = np.arange(test_tunnel_info['xlim'][0], test_tunnel_info['xlim'][1], dx)[:-1]  # ignore the last bound
        vel_bins = [[] for _ in range(len(xs))]
        pos_bins = [[] for _ in range(len(xs))]

        if not trajs:
            # no data
            continue
        for traj in trajs:
            for thingy in traj:
                x, y = thingy['old_pose']['position'][:2]
                xp = thingy['pose']['position'][0]
                if x > test_tunnel_info['xlim'][0]:
                    bindex = np.sum(x > xs) - 1  # subtract 1 to put it on the lower end
                    vel_bins[bindex].append((xp - x)/dt)
                    pos_bins[bindex].append(y)

        plt.plot(xs[[len(b) > 0 for b in vel_bins]],
                 [np.mean(b) for b in vel_bins if len(b) > 0])
        plt.xlabel('x')
        plt.ylabel('avg forward velocity')
        plt.title('avg vel epoch ' + str(epoch))
        plt.xlim(xlim)
        fname = os.path.join(individual_traj_dir,
                             'avg_vel_' +
                             'tunnel_' + str(tunnel_idx) + '_' +
                             'epoch_' + str(epoch) + '.png'
                             )
        # plt.savefig(fname, bbox_inches='tight')
        plt.close()

        ylim = plt_env()
        plt.gca().set_aspect('equal')
        pos_means = np.array([np.mean(b) for b in pos_bins if len(b) > 0])
        pos_std = np.array([np.std(b) for b in pos_bins if len(b) > 0])
        plt.plot(xs[[len(b) > 0 for b in vel_bins]],
                 pos_means,
                 label='averaged y positions ')
        plt.fill_between(xs[[len(b) > 0 for b in vel_bins]],
                         pos_means - pos_std,
                         pos_means + pos_std,
                         alpha=.2,
                         )
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('average y positions epoch ' + str(epoch))
        plt.legend(loc='center left', bbox_to_anchor=(.69, -.35))
        plt.xlim(xlim)
        plt.ylim(ylim)
        fname = os.path.join(individual_traj_dir,
                             'avg_pos_' +
                             'tunnel_' + str(tunnel_idx) + '_' +
                             'epoch_' + str(epoch) + '.png'
                             )
        # plt.savefig(fname, bbox_inches='tight')
        plt.close()

        # fig, ax = plt.subplots()
        plt.xlabel('x axis')
        plt.ylabel('y axis')

        ylim = plt_env()
        plt.gca().set_aspect('equal')

        alpha = 1


        def get_dist_traveled(traj):
            return traj[-1]['pose']['position'][0]
            return traj[-1]['pose']['position'][0] - traj[0]['pose']['position'][0]


        trajs.sort(key=get_dist_traveled)

        plot_stuff = []
        prop_succ = 0
        for traj in trajs:
            kwargs = dict()
            last_info = traj[-1]['info']
            collided = last_info['collided']
            succ = last_info['succ']
            prop_succ += succ/len(trajs)
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

        rwds = np.array([sum(t['reward'] for t in traj) for traj in trajs])

        dists = np.array([get_dist_traveled(traj) for traj in trajs])
        epochs.append(epoch)
        prop_successful.append(prop_succ)
        medians.append(np.median(dists))
        maxes.append(np.max(dists))
        means.append(np.mean(dists))
        mins.append(np.min(dists))
        rwd_medians.append(np.median(rwds))
        rwd_maxes.append(np.max(rwds))
        rwd_means.append(np.mean(rwds))
        rwd_mins.append(np.min(rwds))
        for traj, kwargs in plot_stuff:
            positions = np.stack([traj[0]['old_pose']['position']] +
                                 [dic['pose']['position'] for dic in traj],
                                 axis=0)
            plt.plot(positions[:, 0], positions[:, 1], alpha=alpha, **kwargs)
        plt.plot([], [], color='green', label='successful')
        plt.plot([], [], color='yellow', label='timed out')
        plt.plot([], [], color='red', label='crashed')

        plt.ylim(ylim)

        plt.legend(loc='center left', bbox_to_anchor=(1., .5))
        plt.title('epoch ' + str(epoch))
        plt.xlim(xlim)

        fname = os.path.join(individual_traj_dir,
                             'tunnel_' + str(tunnel_idx) + '_' +
                             'epoch_' + str(epoch) + '_trajectories.png'
                             )
        plt.savefig(fname, bbox_inches='tight')
        plt.close()
        fnames.append(fname)
    fname = os.path.join(plot_dir,
                         'tunnel_' + str(tunnel_idx) + '_' +
                         'trajectories_gifed.gif')
    create_gif(image_paths=fnames,
               output_gif_path=fname,
               duration=200,
               )

    plt.plot(epochs, prop_successful, color='purple')

    plt.xlabel('epochs')
    plt.ylabel('Successful proportion of testing trajectories')
    plt.ylim((0, 1.05))
    plt.title("Proportion of success throughout training")

    plt.savefig(os.path.join(plot_dir, 'tunnel_' + str(tunnel_idx) + '_success.png'),
                bbox_inches='tight')
    plt.close()

    plt.plot(epochs, means, color='blue', label='means')
    plt.plot(epochs, medians, color='orange', label='median')
    plt.plot(epochs, maxes, color='green', label='max')
    plt.plot(epochs, mins, color='red', label='min')
    plt.xlabel('epochs')
    plt.ylabel('distance traveled')
    plt.title("Distance traveled throughout training")

    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'tunnel_' + str(tunnel_idx) + '_distance_summary.png'),
                bbox_inches='tight')
    plt.close()

    plt.plot(epochs, rwd_means, color='blue', label='means')
    plt.plot(epochs, rwd_medians, color='orange', label='median')
    plt.plot(epochs, rwd_maxes, color='green', label='max')
    plt.plot(epochs, rwd_mins, color='red', label='min')
    plt.xlabel('epochs')
    plt.ylabel('reward sum')
    plt.title("Rewards throughout training")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'tunnel_' + str(tunnel_idx) + '_rwd_summary.png'),
                bbox_inches='tight')
    plt.close()
if not args.keep_individuals:
    shutil.rmtree(individual_traj_dir)
