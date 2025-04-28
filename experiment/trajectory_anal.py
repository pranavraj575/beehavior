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
dx = .1
dt = .1
args = PARSER.parse_args()
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
while True:
    fn = os.path.join(traj_dir, 'traj_' + str(i) + '.pkl')
    if not os.path.exists(fn):
        break
    f = open(fn, 'rb')
    epoch_infos.append(pkl.load(f))
    f.close()
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
        'walls': [[(-5, -2), (25, -2)], [(-5, 2), (25, 2)]],
        # list of (center, width, height (for ellipse, x axis, y axis))
        'obs': [((-9230, -22250), (1, 1)),
                ((-8140, -22080), (1, 1))
                ],
    },
    None,
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

    used = False
    for trajs in epoch_infos:
        for traj in trajs:
            if (
                    traj[0]['old_pose']['position'][1] >= initial_wall_y[0] and
                    traj[0]['old_pose']['position'][1] <= initial_wall_y[1]):
                used = True
                break
    if not used:
        continue

    for all_traj in True, False:

        fnames = []
        for epoch, trajs in enumerate(epoch_infos):
            xs = np.arange(test_tunnel_info['xlim'][0], test_tunnel_info['xlim'][1], dx)[:-1] # ignore the last bound
            speed_bins = [[] for _ in range(len(xs))]

            # filter out the trajectories that are tested in this tunnel

            trajs = [traj
                     for traj in trajs if (
                             traj[0]['old_pose']['position'][1] >= initial_wall_y[0] and
                             traj[0]['old_pose']['position'][1] <= initial_wall_y[1])
                     ]
            if not trajs:
                continue
            for traj in trajs:
                for thingy in traj:
                    x = thingy['old_pose']['position'][0]
                    xp = thingy['pose']['position'][0]
                    if x > test_tunnel_info['xlim'][0]:
                        bindex = np.sum(x > xs) - 1 # subtract 1 to put it on the lower end
                        speed_bins[bindex].append((xp - x)/dt)
            plt.plot(xs[[len(b)>0 for b in speed_bins]],
                     [np.mean(b) for b in speed_bins if len(b)>0])
            plt.xlabel('x')
            plt.ylabel('avg forward velocity')
            plt.title('avg vel epoch ' + str(epoch))
            plt.xlim(xlim)

            fname = os.path.join(individual_traj_dir,
                                 'avg_speed_'+
                                 'tunnel_' + str(tunnel_idx) + '_' +
                                 'epoch_' + str(epoch) + '.png'
                                 )
            plt.savefig(fname, bbox_inches='tight')
            plt.close()
            fig, ax = plt.subplots()
            plt.xlabel('x axis')
            plt.ylabel('y axis')
            plt.gca().set_aspect('equal')
            for pt_list in walls:
                ax.plot([wx for (wx, wy) in pt_list],
                        [wy for (wx, wy) in pt_list],
                        color='black', linewidth=4)

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
                    (trajs[-1], {'color': 'purple', 'label': 'best'})
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
            if all_traj:
                plt.plot([], [], color='green', label='successful')
                plt.plot([], [], color='yellow', label='timed out')
                plt.plot([], [], color='red', label='crashed')

            ylim = plt.ylim()
            xbnd = test_tunnel_info['finish_x']
            plt.plot([xbnd, xbnd], ylim,
                     color='black', linewidth=2, linestyle='--'
                     )
            plt.ylim(ylim)

            plt.legend(loc='center left', bbox_to_anchor=(1., .5))
            plt.title('epoch ' + str(epoch))
            plt.xlim(xlim)

            fname = os.path.join(individual_traj_dir,
                                 'tunnel_' + str(tunnel_idx) + '_' +
                                 ('all_' if all_traj else '') +
                                 'epoch_' + str(epoch) + '_trajectories.png'
                                 )
            plt.savefig(fname, bbox_inches='tight')
            plt.close()
            fnames.append(fname)
        fname = os.path.join(plot_dir,
                             'tunnel_' + str(tunnel_idx) + '_' +
                             ('all_' if all_traj else '') +
                             'traj_summary_gifed.gif')
        create_gif(image_paths=fnames,
                   output_gif_path=fname,
                   duration=200,
                   )

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
    plt.savefig(os.path.join(plot_dir, 'tunnel_' + str(tunnel_idx) + '_' + 'trajectory_summary.png'))
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
    plt.savefig(os.path.join(plot_dir, 'tunnel_' + str(tunnel_idx) + '_' + 'rwd_summary.png'))
    plt.close()
if not args.keep_individuals:
    shutil.rmtree(individual_traj_dir)
