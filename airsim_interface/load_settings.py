import os, sys, ast, json

import airsim

DIR = os.path.dirname(__file__)
SETT_DIR = os.path.join(DIR, 'settings.txt')


class Keychain:
    """keys for useful objects"""
    UE4loc = 'unreal engine'
    Defaultproj = 'defalut project'
    CameraSettingLoc = 'camera settings'


# default locations of these files
default_settings = {
    Keychain.UE4loc: os.path.join('/home', 'pravna',
                                  'UnrealEngine', 'Engine', 'Binaries', 'Linux', 'UE4Editor'),
    Keychain.Defaultproj: os.path.join('/home', 'pravna',
                                       'AirSim', 'Unreal', 'Environments', 'Blocks_4.27', 'Blocks.uproject'),
    Keychain.CameraSettingLoc: os.path.join('/home', 'pravna',
                                            'Documents', 'AirSim', 'settings.json'),
}


def get_camera_settings(sett):
    """
    gets camera settings from settings
    """
    f = open(sett[Keychain.CameraSettingLoc], 'r')
    camera_sett = json.load(f)
    f.close()
    return camera_sett


def get_fov(camera_settings, camera_name='front', image_type=airsim.ImageType.DepthPerspective):
    """
    gets FOV from camera settings
    """
    dic = camera_settings['Vehicles']['Drone0']['Cameras'][camera_name]['CaptureSettings']
    cam_dic=None
    for d in dic:
        if d['ImageType']==image_type:
            cam_dic=d
    if cam_dic is None:
        raise Exception('image type',image_type,'not found')
    return cam_dic['FOV_Degrees']


def get_settings(old_settings=None):
    """
    gets settings, defaults to default_settings
    updates old_settings with whatever is in the settings file
    """
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
    """
    saves settings to SETT_DIR file
    """
    f = open(SETT_DIR, 'w')
    f.write(unparse(sett))
    f.close()


def unparse(itm):
    """
    puts item into readable string to save as .txt
    """
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
    save_settings(get_settings())
    print(get_camera_settings(get_settings())['Vehicles']['Drone0']['Cameras']['front']['CaptureSettings'])
    print(get_fov(camera_settings=get_camera_settings(get_settings())))
