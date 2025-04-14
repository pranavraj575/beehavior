"""
similar to airsim_interface/keyboard_test but with gym envionrnment
"""

if __name__ == '__main__':
    from threading import Thread
    import numpy as np
    import gymnasium as gym
    import argparse

    from curtsies import Input
    import beehavior
    from beehavior.envs.beese_class import BeeseClass

    PARSER = argparse.ArgumentParser(
        description='Control gym enviornment drone with keyboard: '
                    'keys 1234567890 control thrust (1 is least and 0 is most); '
                    'arrow keys control roll/pitch; '
                    'space bar progresses simulation for a quarter second and pauses; '
                    'c clears roll/pitch; '
                    'r to reset simulation; '
                    'Q (shift + q) to stop python script'
    )

    PARSER.add_argument("--env", action='store', required=False, default='Beese-v0',
                        choices=('Beese-v0', 'HiBee-v0', 'ForwardBee-v0'),
                        help="RL gym class to run")
    PARSER.add_argument("--dt", type=float, required=False, default=.25,
                        help="time in between commands sent to simulation")
    PARSER.add_argument("--radian-ctrl", type=float, required=False, default=np.pi/18,
                        help="radians that each arrow command changes roll/pitch")
    PARSER.add_argument("--max-ctrl", type=int, required=False, default=5,
                        help="number of times you can increment by radian-ctrl")
    PARSER.add_argument("--thrust-n", type=int, required=False, default=10, choices=list(range(2, 11)),
                        help="number of potential thrust values, between 2 and 10")
    PARSER.add_argument('--real-time', action='store_true', required=False,
                        help='whether to run simulation continuously, default is to pause ever dt seconds')
    args = PARSER.parse_args()

    discrete = list('1234567890')[:args.thrust_n]

    thrust = .6
    lr = 0  # whether left key or right key is being held
    bf = 0
    none_step = False

    reset = False
    close = False


    def get_cmd():
        global thrust, lr, bf
        # number of time right is pressed-number of time left is pressed
        return np.array([lr*args.radian_ctrl, bf*args.radian_ctrl, thrust])


    def record():
        global bf, lr, thrust, none_step, reset, close
        with Input(keynames='curses') as input_generator:
            for e in input_generator:
                k = repr(e).replace("'", '')
                if k == 'KEY_UP':
                    bf = min(1 + bf, args.max_ctrl)
                if k == 'KEY_DOWN':
                    bf = max(bf - 1, -args.max_ctrl)
                if k == 'KEY_LEFT':
                    lr = max(lr - 1, -args.max_ctrl)
                if k == 'KEY_RIGHT':
                    lr = min(1 + lr, args.max_ctrl)
                if k == 'c':
                    lr = 0
                    bf = 0
                if k == ' ':
                    none_step = True
                if k in discrete:
                    thrust = discrete.index(k)/(args.thrust_n - 1)
                if k == 'r':
                    reset = True
                if k == 'Q':
                    close = True
                    return


    th = Thread(target=record, daemon=True)
    th.start()

    env = gym.make(args.env,
                   dt=args.dt,
                   action_bounds=args.radian_ctrl*args.max_ctrl,
                   real_time=args.real_time,
                   action_type=BeeseClass.ACTION_ROLL_PITCH_YAW,
                   )

    env.reset()

    reward = 0
    strout_old = ''

    while not close:
        cmd = get_cmd()
        if none_step or args.real_time:
            none_step = False
            observation, reward, termination, truncation, info = env.step(action=cmd)
            if termination or truncation:
                reset = True
        strout = ('\033[2K' +
                  str(tuple(zip(['r:', 'p:', 'thrust:'], cmd))) +
                  ' last rwd:' + str(reward) + '                  \r')
        if strout != strout_old:
            print(strout, end='')
            strout_old = strout
        if reset:
            print('resetting')
            env.reset()

            reset = False

            thrust = .6
            lr = 0  # whether left key or right key is being held
            bf = 0
            none_step = False
    env.close()
