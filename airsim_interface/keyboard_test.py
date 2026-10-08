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
import airsim,os

def add_gaussiannoise(of, noise):
    return of+np.random.normal(0,noise,of.shape)
def apply_subsample(of,ss):
    temp=of[...,::ss[0],::ss[1]]
    for i in range(ss[0]):
        for j in range(ss[1]):
            of[...,i::ss[0],j::ss[1]]=temp

    return of
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
                    'p to display pose data; '
                    'Q (shift + q) to stop python script'
    )

    PARSER.add_argument("--dt", type=float, required=False, default=.25,
                        help="time in between commands sent to simulation")
    PARSER.add_argument("--radian-ctrl", type=float, required=False, default=np.pi/18,
                        help="radians that each arrow command changes roll/pitch")
    PARSER.add_argument("--max-ctrl", type=int, required=False, default=9,
                        help="number of times you can increment by radian-ctrl")
    PARSER.add_argument("--imgs", type=str, required=False, default=None,
                        help="dir to store images")
    PARSER.add_argument("--thrust-n", type=int, required=False, default=10, choices=list(range(2, 11)),
                        help="number of potential thrust values, between 2 and 10")
    PARSER.add_argument('--real-time', action='store_true', required=False,
                        help='whether to run simulation continuously, default is to pause ever dt seconds')
    PARSER.add_argument('--without-game-interface', action='store_true', required=False,
                        help='do not connect to unreal engine, just print out the cmd values')
    args = PARSER.parse_args()
    game_interface = not args.without_game_interface

    if game_interface:
        from airsim_interface.interface import step, connect_client, disconnect_client, get_depth_img, of_geo_from_client

    discrete = list('1234567890')[:args.thrust_n]

    thrust = .6
    lr = 0  # whether left key or right key is being held
    bf = 0
    none_step = False

    reset = False
    close = False
    img = False
    pose_data = False


    def get_cmd():
        global thrust, lr, bf
        # number of time right is pressed-number of time left is pressed
        return lr*args.radian_ctrl, bf*args.radian_ctrl, thrust


    def record():
        global bf, lr, thrust, none_step, reset, close, img, pose_data
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
                if k == 'p':
                    pose_data = True
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

        if pose_data and game_interface:
            pose = client.simGetVehiclePose()
            print(pose)
            pose_data = False
        if img and game_interface:
            if args.imgs:
                os.makedirs(args.imgs,exist_ok=True)
            from matplotlib import pyplot as plt

            of = of_geo_from_client(client=client,
                                    camera_name='front',
                                    vehicle_name='',
                                    FOVx_degrees=None,
                                    ignore_angular_velocity=True,
                                    )
            response = client.simGetImages([airsim.ImageRequest('front', airsim.ImageType.Scene, False, False)])
            img_data = response[0].image_data_uint8
            if img_data:
                image = np.frombuffer(img_data, dtype=np.uint8).reshape(response[0].height, response[0].width, 3)
                # image = cv2.resize(image, (320, 240))  #(1080, 720) : Resize to 320x240 for performance

                plt.imshow(image[:, :, ::-1], interpolation='nearest', cmap="coolwarm")
                if args.imgs:
                    plt.savefig(os.path.join(args.imgs,'test.png'))
                else:
                    plt.show()
                plt.close()
            of = np.transpose(of, axes=(1, 2, 0))


            def disp(thingy, title=None,save=None):
                temp = np.zeros((*of.shape[:2], 3), dtype=np.uint8)

                mx = np.max(thingy)
                mn = np.min(thingy)
                if mx == mn:
                    thingy = np.zeros_like(thingy)
                else:
                    thingy = (thingy - mn)/(mx - mn)*255  # to byte

                temp[:, :, :] = np.ndarray.astype(thingy, dtype=np.uint8)

                plt.imshow(temp, interpolation='nearest', cmap="coolwarm" )

                def fmt(num):
                    if abs(num) >= 1 and abs(num) <= 1000:
                        return '{:.2f}'.format(num)
                    if abs(num) > 1000:
                        return '{:.0f}'.format(num)
                    return '{:.2E}'.format(num)

                plt.title('low: ' + fmt(mn) + '; high: ' + fmt(mx))
                plt.axis('off')
                if title is not None:
                    plt.suptitle(title)

                plt.tight_layout()
                if save:
                    plt.savefig(save)
                else:
                    plt.show()
                plt.close()


            for dim in range(2):
                if args.imgs:
                    save= os.path.join(args.imgs,f'dim_{dim}.png')
                else:
                    save=None
                disp(np.abs(of)[:, :, dim:dim + 1],save=save)

            for value, title in (
                    (np.linalg.norm(of, axis=-1, keepdims=True), 'Raw Optic Flow'),
                    (apply_subsample(np.linalg.norm(of, axis=-1, keepdims=True),[2,2]), 'Subsampled Optic Flow (2)'),
                    (apply_subsample(np.linalg.norm(of, axis=-1, keepdims=True),[4,4]), 'Subsampled Optic Flow (4)'),
                    (add_gaussiannoise(np.linalg.norm(of, axis=-1, keepdims=True),10), 'Noisy Optic Flow'),
                    (np.log(np.clip(np.linalg.norm(of, axis=-1, keepdims=True), 10e-10, np.inf)), 'Log Optic Flow'),
                    # (lambda x: np.clip(np.log(x),-1,np.inf), 'Clipped Log Optic Flow'),
                    (np.sqrt(np.linalg.norm(of, axis=-1, keepdims=True)), 'Sqrt Optic Flow'),
            ):
                if args.imgs:
                    save=os.path.join(args.imgs,title.replace(' ','_')+'.png')
                else:
                    save=None
                disp(value, title=title,save=save)

            if img_data:
                image = np.frombuffer(img_data, dtype=np.uint8).reshape(response[0].height, response[0].width, 3)
                # image = cv2.resize(image, (320, 240))  #(1080, 720) : Resize to 320x240 for performance

                plt.imshow(image[:, :, ::-1], interpolation='nearest', cmap='coolwarm')
                h, w = np.meshgrid(np.arange(of.shape[0]), np.arange(of.shape[1]))
                ss = 10
                of_disp = np.transpose(of, axes=(1, 0, 2))
                # inverted from image (height is top down) to np plot (y dim  bottom up)
                plt.quiver(w[::ss, ::ss], h[::ss, ::ss],
                           of_disp[::ss, ::ss, 0],
                           -of_disp[::ss, ::ss, 1],
                           color='black',
                           )
                if args.imgs:
                    plt.savefig(os.path.join(args.imgs, f'quiver.png'))
                else:
                    plt.show()
                plt.close()
            img = False
    if game_interface:
        disconnect_client(client=client)
