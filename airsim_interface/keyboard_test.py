import time

if __name__ == '__main__':
    from pynput.keyboard import Key, Listener

    thrust_n = 10
    discrete = list('1234567890')[:thrust_n]

    thrust = 0
    close = False
    lr = [False, False]  # whether left key or right key is being held
    bf = [False, False]


    def on_press(key):
        global thrust, lr, bf, close

        if 'char' in dir(key) and key.char in discrete:
            thrust = discrete.index(key.char)/(thrust_n - 1)

        elif key == Key.down:
            bf[0] = True
        elif key == Key.up:
            bf[1] = True

        elif key == Key.left:
            lr[0] = True
        elif key == Key.right:
            lr[1] = True

        elif key == Key.esc:
            close = True


    def on_release(key):
        global thrust, lr, bf

        if key == Key.down:
            bf[0] = False
        elif key == Key.up:
            bf[1] = False

        elif key == Key.left:
            lr[0] = False
        elif key == Key.right:
            lr[1] = False


    def get_cmd():
        global thrust, lr, bf
        x = -1*lr[0] + lr[1]
        y = -1*bf[0] + bf[
            1]  # -1, 0, or 1, depending if (just down is held), (either both held or none held), (just up held)
        return thrust, x, y


    hear = Listener(on_press=on_press, on_release=on_release)
    hear.start()
    old_cmd=None
    while not close:
        cmd=get_cmd()
        if cmd!=old_cmd:
            print(*zip(['thrust:', 'x:', 'y:'], cmd))
        old_cmd=cmd
        time.sleep(.01)
