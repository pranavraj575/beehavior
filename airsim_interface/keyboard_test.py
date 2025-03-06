"""
Basic keyboard controller for quadrotor
start a quadcopter project in game+windowed mode `<...>/Engine/Binaries/Linux/UE4Editor <...>/AirSim/Unreal/Environments/Blocks_4.27/Blocks.uproject -game -windowed`
in another terminal, run `python3 airsim_interface/keyboard_test.py`
* control the drone!
  * keys 1234567890 control thrust, 1 is least and 0 is most
  * arrow keys control roll/pitch
  * each command runs the simulation for a quarter second and pauses
  * b progresses simulation without submitting an action
  * space bar clears roll/pitch and progresses simulation
  * r to reset simulation
  * Q (shift + q) to stop python script
"""

from threading import Thread

import numpy as np

if __name__ == '__main__':
    from curtsies import Input
    from airsim_interface.interface import step, connect_client, disconnect_client

    dt = .25
    radian_ctrl=np.pi/36

    thrust_n = 10

    discrete = list('1234567890')[:thrust_n]

    thrust = 0
    lr = [0, 0]  # whether left key or right key is being held
    bf = [0, 0]
    none_step = False

    reset = False
    close = False

    game_interface = True


    def get_cmd():
        global thrust, lr, bf
        x = -1*lr[0] + lr[1]
        y = -1*bf[0] + bf[
            1]  # -1, 0, or 1, depending if (just down is held), (either both held or none held), (just up held)
        return thrust, x*radian_ctrl, y*radian_ctrl


    def record():
        global bf, lr, thrust, none_step, reset, close
        with Input(keynames='curses') as input_generator:
            for e in input_generator:
                k = repr(e).replace("'", '')
                if k == 'KEY_UP':
                    bf[1] +=1
                    bf[0] -=1
                if k == 'KEY_DOWN':
                    bf[0] += 1
                    bf[1] -= 1
                if k == 'KEY_LEFT':
                    lr[0] += 1
                    lr[1] -= 1
                if k == 'KEY_RIGHT':
                    lr[1] += 1
                    lr[0] -= 1
                if k == ' ':
                    lr = [0, 0]
                    bf = [0, 0]
                    none_step = True
                if k == 'b':
                    none_step = True
                if k in discrete:
                    thrust = discrete.index(k)/(thrust_n - 1)
                if k == 'r':
                    reset = True
                if k == 'Q':
                    close = True


    client = None
    th = Thread(target=record, daemon=True)
    th.start()
    if game_interface:
        client = connect_client()
    old_cmd = get_cmd()
    while not close:
        cmd = get_cmd()

        if cmd != old_cmd or none_step:
            none_step = False
            print(*zip(['thrust:', 'x:', 'y:'], cmd))
            thrust, x, y = cmd
            if game_interface:
                step(client=client,
                     seconds=dt,
                     cmd=lambda: client.moveByRollPitchYawrateThrottleAsync(roll=x,
                                                                            pitch=y,
                                                                            yaw_rate=0,
                                                                            throttle=thrust,
                                                                            duration=1),
                     pause_after=True,
                     )
        old_cmd = cmd
        if reset:
            print('resetting')
            if game_interface:
                client.reset()
                connect_client(client=client)
            old_cmd = get_cmd()
            reset = False

            thrust = 0
            lr = [0, 0]  # whether left key or right key is being held
            bf = [0, 0]
            none_step = False

    disconnect_client(client=client)
