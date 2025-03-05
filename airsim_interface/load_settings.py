import os, sys, ast

DIR = os.path.dirname(os.path.join(os.getcwd(), sys.argv[0]))
SETT_DIR = os.path.join(DIR, 'settings.txt')


class Keychain:
    UE4loc = 'unreal engine'
    Defaultproj = 'defalut project'


default_settings = {
    Keychain.UE4loc: os.path.join('/home', 'pravna',
                                  'UnrealEngine', 'Engine', 'Binaries', 'Linux', 'UE4Editor'),
    Keychain.Defaultproj: os.path.join('/home', 'pravna',
                                       'AirSim', 'Unreal', 'Environments', 'Blocks_4.27', 'Blocks.uproject'),
}


def get_settings(old_settings=None):
    if old_settings is None:
        old_settings = default_settings
    if not os.path.exists(SETT_DIR):
        save_settings(default_settings)
    f = open(SETT_DIR, 'r')
    read_sett = ast.literal_eval(f.read())
    f.close()
    old_settings.update(read_sett)
    return old_settings


def save_settings(sett):
    f = open(SETT_DIR, 'w')
    f.write(unparse(sett))
    f.close()


def unparse(itm):
    if type(itm) == dict:
        s = '{\n'
        for k, v in itm.items():
            s += unparse(k) + ':' + unparse(v) + ',\n'
        s += '}\n'
    elif type(itm) == list:
        s = '[\n'
        for thing in itm:
            s += unparse(thing) + ',\n'
        s += ']\n'
    elif type(itm) == str:
        s = '"' + itm.replace('"', '\\"') + '"'
    else:
        s = str(itm)
    return s


if __name__ == '__main__':
    print(get_settings())
