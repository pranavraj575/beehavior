import os, shutil
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import argparse
from PIL import Image

import cv2
import skvideo.io


def create_mp4(image_paths, output_mp4_path, duration=200, debug=False):
    writer = skvideo.io.FFmpegWriter(output_mp4_path, outputdict={
        '-vcodec': 'libx264',  # use the h.264 codec
        '-crf': '0',  # set the constant rate factor to 0, which is lossless
        # '-framerate':str(1), # frequency, inverse of duration
        '-preset': 'veryslow'  # the slower the better compression, in princple, try
        # other options see https://trac.ffmpeg.org/wiki/Encode/H.264
    })
    default_fps = 25
    duration = duration/1000  # convert to seconds/image
    time_elapsed = 0
    while time_elapsed < duration*len(image_paths):
        # time_elapsed//duration<= time_elapsed/duration<len(image_paths)
        image_path = image_paths[int(time_elapsed//duration)]
        img = cv2.imread(filename=image_path)
        writer.writeFrame(img[:, :, ::-1])
        time_elapsed += 1/default_fps
        if debug:
            print('wrote', time_elapsed, 'of', duration*len(image_paths), end='\r')
    if debug:
        print('wrote video of length', time_elapsed - 1/default_fps)
    writer.close()
    cv2.destroyAllWindows()


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


if __name__ == '__main__':
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
    PARSER.add_argument('--alpha-traj', type=float, required=False, default=.69,
                        help='opacity of trajectory lines')

    PARSER.add_argument('--remove-axis-ticks', action='store_true', required=False,
                        help='dont plot values for x and y axis')

    PARSER.add_argument('--no-legend', action='store_true', required=False,
                        help='dont plot legend')

    PARSER.add_argument('--necessary-leg', action='store_true', required=False,
                        help='dont plot things in legend that arent in plot')
    PARSER.add_argument('--take', type=int, required=False, default=None,
                        help='take this many trajs')
    PARSER.add_argument('--dpi', type=int, required=False, default=100,
                        help='dpi for saved images')
    PARSER.add_argument('--fontsize', type=int, required=False, default=None,
                        help='font size for saved images')
    PARSER.add_argument('--inversion', action='store_true', required=False,
                        help='plot y axis inverted')

    args = PARSER.parse_args()
    if args.fontsize is not None:
        plt.rcParams.update({'font.size': args.fontsize})
    yinversion = -1 if args.inversion else 1
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
            # list of vertex lists
            'walls': [[(-5, -8.35), (1.05, -8.35), (6.55, -6.9), (12.25, -8.35), (18.05, -6.9), (23.75, -8.35)],
                      [(-5, -3.15), (1.05, -3.15), (6.55, -4.6), (12.25, -3.15), (18.05, -4.6), (23.75, -3.15)]],
            'xlim': (-3, 21),
            # list of (center, width, height (for ellipse, x axis, y axis))
            'obs': [],
        },
        # basic default tunnel with two obstacles
        {
            'walls': [[(-5, -2.1), (25, -2.1)], [(-5, 2.1), (25, 2.1)]],
            'obs': [((-9230, -22250), (1, 1)),
                    ((-8140, -22080), (1, 1))
                    ],
        },
        # tunnel with one center obstacle and two further obstacles
        {
            'walls': [[(-5, 2.9), (25, 2.9)], [(-5, 8.8), (25, 8.8)], ],
            'obs': [
                ((-9230, -21530), (1, 1)),
                ((-8030, -21300), (1, 1)),
                ((-7900, -21790), (1, 1)),
            ],
        },
        # narrow tunnel
        {
            'walls': [[(-5, 9.75), (25, 9.75)], [(-5, 11.85), (25, 11.85)], ],
            'obs': [
                ((-8620, -20980), (1, 1)),
                ((-7820, -21130), (1, .5)),
            ],
        },
        # tunnel with two side obstacles then one center obstacle
        {
            'walls': [[(-5, 12.55), (25, 12.55)], [(-5, 16.5), (25, 16.5)], ],
            'obs': [
                ((-8830, -20860), (1, 1.5)),
                ((-8720, -20510), (1, 1.5)),
                ((-8020, -20680), (1, 1)),
            ],
        },
        # wide tunnel, one center then two side obstacles
        {
            'walls': [[(-5, 17.69), (25, 17.69)], [(-5, 26.8), (25, 26.8)], ],
            'obs': [
                ((-8760, -19840), (3, 3)),
                ((-7900, -20250), (1, 1.5)),
                ((-8060, -19480), (1, 1.5)),
            ],
        },
        # side to side walls tunnel
        {
            'walls': [[(-5, 28),
                       (8.5 + (-2.21 - 1.9), 28), (8.5, 31.88), (8.5, 28),
                       (25, 28)
                       ],
                      [(-5, 35.3),
                       (-2.21, 35.3), (1.9, 31.5), (1.9, 35.3),
                       (-2.21 + 13.5, 35.3), (1.9 + 13.5, 31.5), (1.9 + 13.5, 35.3),
                       (25, 35.3)
                       ],
                      ],
            'obs': [],
        },
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
            walled = False
            for pt_list in walls:
                ax.plot([wx for (wx, wy) in pt_list],
                        [yinversion*wy for (wx, wy) in pt_list],
                        color='black', linewidth=4, label='wall' if not walled else None)
                walled = True
            ylim = plt.ylim()  # y limits should be bounded by walls

            xbnd = test_tunnel_info['finish_x']
            ax.plot([xbnd, xbnd], plt.ylim(),
                    color='black', linewidth=2, linestyle='--', label='goal'
                    )

            obstacled = False
            for effective_bnd in (False, True):
                alpha = [1, .5][int(effective_bnd)]
                for c, (w, h) in test_tunnel_info['obs']:
                    if effective_bnd:
                        # drone has a radius of about 1m, so obstacles need to be elongated by a radius of .5,
                        #   representing a crash if drone COM is within this area
                        # thus, increase width and height (diameter) by 1 each

                        w += 1
                        h += 1
                    c = (c[0] + 9390)/100, yinversion*(c[1] + 22145)/100
                    # ax.add_patch(plt.Circle(c, r, alpha=alpha))
                    ax.add_patch(Ellipse(xy=c,
                                         width=w,
                                         height=h,
                                         angle=0,
                                         alpha=alpha,
                                         label='obstacle' if (effective_bnd and (not obstacled)) else None,
                                         ))
                    if effective_bnd:
                        obstacled = True

            if args.remove_axis_ticks:
                ax.set_xticklabels([])
                ax.set_xticks([])

                ax.set_yticklabels([])
                ax.set_yticks([])
            return ylim


        fnames = []
        for epoch, trajs in enumerate(relevant_epoch_infos):
            xs = np.arange(test_tunnel_info['xlim'][0], test_tunnel_info['xlim'][1], dx)[:-1]  # ignore the last bound
            vel_bins = [[] for _ in range(len(xs))]
            pos_bins = [[] for _ in range(len(xs))]

            if not trajs:
                # no data
                continue

            if args.take is not None and len(trajs) > args.take:
                trajs = trajs[:args.take]
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
            pos_means = np.array([yinversion*np.mean(b) for b in pos_bins if len(b) > 0])
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
            if not args.no_legend: plt.legend(loc='center left', bbox_to_anchor=(.69, -.35))
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


            def get_dist_traveled(traj):
                return traj[-1]['pose']['position'][0]
                return traj[-1]['pose']['position'][0] - traj[0]['pose']['position'][0]


            trajs.sort(key=get_dist_traveled)

            plot_stuff = []
            cnt = 0
            cnt_succ = 0
            cnt_crashed = 0
            cnt_timeout = 0
            for traj_idx, traj in enumerate(trajs):
                cnt += 1
                kwargs = dict()
                last_info = traj[-1]['info']
                collided = last_info['collided']
                succ = last_info.get('succ', not collided)
                rewards = np.array([dic['reward'] for dic in traj])
                kwargs['alpha'] = args.alpha_traj
                if succ:
                    kwargs['color'] = 'green'
                    kwargs['zorder'] = 3
                    cnt_succ += 1
                elif not collided:
                    kwargs['color'] = 'yellow'
                    kwargs['zorder'] = 2
                    cnt_timeout += 1
                else:
                    kwargs['color'] = 'red'
                    kwargs['zorder'] = 4
                    cnt_crashed += 1
                plot_stuff.append((traj, kwargs))

            rwds = np.array([sum(t['reward'] for t in traj) for traj in trajs])

            dists = np.array([get_dist_traveled(traj) for traj in trajs])
            epochs.append(epoch)
            prop_successful.append(cnt_succ/cnt)
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
                plt.plot(positions[:, 0], yinversion*positions[:, 1], **kwargs)
            if args.necessary_leg:
                if cnt_succ: plt.plot([], [], color='green', label='successful')
                if cnt_timeout: plt.plot([], [], color='yellow', label='timed out')
                if cnt_crashed: plt.plot([], [], color='red', label='crashed')
            else:
                plt.plot([], [], color='green', label='successful')
                plt.plot([], [], color='yellow', label='timed out')
                plt.plot([], [], color='red', label='crashed')

            plt.ylim(ylim)

            if not args.no_legend: plt.legend(loc='center left', bbox_to_anchor=(1., .5))
            plt.title('epoch ' + str(epoch))
            plt.xlim(xlim)

            fname = os.path.join(individual_traj_dir,
                                 'tunnel_' + str(tunnel_idx) + '_' +
                                 'epoch_' + str(epoch) + '_trajectories' + (
                                     '_no_leg' if args.no_legend else '') + '.png'
                                 )
            plt.savefig(fname, bbox_inches='tight', dpi=args.dpi)
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
        plt.ylabel('Successful proportion')
        plt.ylim((0, 1.05))
        plt.title("Successful test trajectories throughout training")
        fig = plt.gcf()
        width, height = fig.get_size_inches()
        fig.set_size_inches(width, height*.420)
        plt.savefig(os.path.join(plot_dir, 'tunnel_' + str(tunnel_idx) + '_success.png'),
                    bbox_inches='tight', dpi=args.dpi)
        plt.close()

        plt.plot(epochs, means, color='blue', label='means')
        plt.plot(epochs, medians, color='orange', label='median')
        plt.plot(epochs, maxes, color='green', label='max')
        plt.plot(epochs, mins, color='red', label='min')
        plt.xlabel('epochs')
        plt.ylabel('distance traveled')
        plt.title("Distance traveled throughout training")

        if not args.no_legend: plt.legend()
        plt.savefig(os.path.join(plot_dir, 'tunnel_' + str(tunnel_idx) + '_distance_summary.png'),
                    bbox_inches='tight', dpi=args.dpi)
        plt.close()

        plt.plot(epochs, rwd_means, color='blue', label='means')
        plt.plot(epochs, rwd_medians, color='orange', label='median')
        plt.plot(epochs, rwd_maxes, color='green', label='max')
        plt.plot(epochs, rwd_mins, color='red', label='min')
        plt.xlabel('epochs')
        plt.ylabel('reward sum')
        plt.title("Rewards throughout training")
        if not args.no_legend: plt.legend()
        plt.savefig(os.path.join(plot_dir, 'tunnel_' + str(tunnel_idx) + '_rwd_summary.png'),
                    bbox_inches='tight', dpi=args.dpi)
        plt.close()
    if not args.keep_individuals:
        shutil.rmtree(individual_traj_dir)
