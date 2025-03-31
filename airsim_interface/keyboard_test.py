"""
Basic keyboard controller for quadrotor
start a quadcopter project in game+windowed mode `<...>/Engine/Binaries/Linux/UE4Editor <...>/AirSim/Unreal/Environments/Blocks_4.27/Blocks.uproject -game -windowed`
in another terminal, run `python3 airsim_interface/keyboard_test.py`
* control the drone!
  * keys 1234567890 control thrust, 1 is least and 0 is most
  * arrow keys control roll/pitch
  * space bar progresses simulation for a quarter second and pauses
  * c clears roll/pitch
  * r to reset simulation
  * i to display images
  * Q (shift + q) to stop python script
"""
import airsim

if __name__ == '__main__':
    from threading import Thread
    import numpy as np
    import argparse
    from curtsies import Input

    PARSER = argparse.ArgumentParser(
        description='Control gym enviornment drone with keyboard: '
                    'keys 1234567890 control thrust (1 is least and 0 is most); '
                    'arrow keys control roll/pitch; '
                    'space bar progresses simulation for a quarter second and pauses; '
                    'c clears roll/pitch; '
                    'r to reset simulation; '
                    'i to display camera images; '
                    'Q (shift + q) to stop python script'
    )

    PARSER.add_argument("--dt", type=float, required=False, default=.25,
                        help="time in between commands sent to simulation")
    PARSER.add_argument("--radian-ctrl", type=float, required=False, default=np.pi/18,
                        help="radians that each arrow command changes roll/pitch")
    PARSER.add_argument("--max-ctrl", type=int, required=False, default=9,
                        help="number of times you can increment by radian-ctrl")
    PARSER.add_argument("--thrust-n", type=int, required=False, default=10, choices=list(range(2, 11)),
                        help="number of potential thrust values, between 2 and 10")
    PARSER.add_argument('--real-time', action='store_true', required=False,
                        help='whether to run simulation continuously, default is to pause ever dt seconds')
    PARSER.add_argument('--without-game-interface', action='store_true', required=False,
                        help='do not connect to unreal engine, just print out the cmd values')
    args = PARSER.parse_args()
    game_interface = not args.without_game_interface

    if game_interface:
        from airsim_interface.interface import step, connect_client, disconnect_client, get_depth_img, of_geo

    discrete = list('1234567890')[:args.thrust_n]

    thrust = .6
    lr = 0  # whether left key or right key is being held
    bf = 0
    none_step = False

    reset = False
    close = False
    img = False


    def get_cmd():
        global thrust, lr, bf
        # number of time right is pressed-number of time left is pressed
        return lr*args.radian_ctrl, bf*args.radian_ctrl, thrust


    def record():
        global bf, lr, thrust, none_step, reset, close, img
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
                if k == 'i':
                    img = True
                if k == 'Q':
                    close = True
                    return


    th = Thread(target=record, daemon=True)
    th.start()

    if game_interface:
        client = connect_client()
    else:
        client = None

    old_strout = ''
    while not close:
        cmd = get_cmd()

        strout = ('\033[2K' +
                  str(tuple(zip(['r:', 'p:', 'thrust:'], cmd))) +
                  '                  \r')
        if strout != old_strout:
            print(strout, end='')
            old_strout = strout
        if none_step or args.real_time:
            none_step = False
            x, y, thrust = cmd
            if game_interface:
                step(client=client,
                     seconds=args.dt,
                     cmd=lambda: client.moveByRollPitchYawrateThrottleAsync(roll=x,
                                                                            pitch=y,
                                                                            yaw_rate=0,
                                                                            throttle=thrust,
                                                                            duration=1),
                     pause_after=not args.real_time,
                     )
            else:
                print('\033[2Ksent:', *zip(['thrust:', 'x:', 'y:'], cmd), end='\r')
        if reset:
            print('resetting')
            if game_interface:
                client.reset()
                connect_client(client=client)
            reset = False

            thrust = .6
            lr = 0  # whether left key or right key is being held
            bf = 0
            none_step = False
        if img and game_interface:
            from matplotlib import pyplot as plt

            response = client.simGetImages([airsim.ImageRequest('front', airsim.ImageType.Scene, False, False)])
            img_data = response[0].image_data_uint8
            if img_data:
                image = np.frombuffer(img_data, dtype=np.uint8).reshape(response[0].height, response[0].width, 3)
                # image = cv2.resize(image, (320, 240))  #(1080, 720) : Resize to 320x240 for performance

                plt.imshow(image[:, :, ::-1], interpolation='nearest', )
                plt.show()
            plt.close()
            of = of_geo(client=client, camera_name='front', vehicle_name='', FOVx=None)
            of = np.transpose(of, axes=(1, 2, 0))


            def disp(thingy, title=None):
                temp = np.zeros((*of.shape[:2], 3), dtype=np.uint8)

                mx = np.max(thingy)
                mn = np.min(thingy)
                if mx == mn:
                    thingy = np.zeros_like(thingy)
                else:
                    thingy = (thingy - mn)/(mx - mn)*255  # to byte

                temp[:, :, :] = np.ndarray.astype(thingy, dtype=np.uint8)

                plt.imshow(temp, interpolation='nearest', )

                def fmt(num):
                    if abs(num) >= 1 and abs(num) <= 1000:
                        return '{:.2f}'.format(num)
                    if abs(num) > 1000:
                        return '{:.0f}'.format(num)
                    return '{:.2E}'.format(num)

                plt.title('black: ' + fmt(mn) + '; white: ' + fmt(mx))
                if title is not None:
                    plt.suptitle(title)

                plt.tight_layout()
                plt.show()


            for dim in range(2):
                disp(np.abs(of)[:, :, dim:dim + 1])

            for mapping, title in (
                    (lambda x: x, 'Raw Optic Flow'),
                    (lambda x: np.log(np.clip(x, 10e-10, np.inf)), 'Log Optic Flow'),
                    # (lambda x: np.clip(np.log(x),-1,np.inf), 'Clipped Log Optic Flow'),
                    (np.sqrt, 'Sqrt Optic Flow'),
            ):
                disp(mapping(np.linalg.norm(of, axis=-1, keepdims=True)), title=title)

            if img_data:
                image = np.frombuffer(img_data, dtype=np.uint8).reshape(response[0].height, response[0].width, 3)
                # image = cv2.resize(image, (320, 240))  #(1080, 720) : Resize to 320x240 for performance

                plt.imshow(image[:, :, ::-1], interpolation='nearest', )
                h, w = np.meshgrid(np.arange(of.shape[0]), np.arange(of.shape[1]))
                ss = 10
                of_disp = np.transpose(of, axes=(1, 0, 2))  # inverted from image (height is top down) to np plot (y dim  bottom up)

                plt.quiver(w[::ss, ::ss], h[::ss, ::ss],
                           of_disp[::ss, ::ss, 0],
                           -of_disp[::ss, ::ss, 1],
                           color='red',
                           )
                plt.show()
            img = False
    if game_interface:
        disconnect_client(client=client)
