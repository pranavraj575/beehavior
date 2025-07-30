# Understanding visual attention beehind bee-inspired UAV navigation

Training a quadrotor drone through RL with a bio-inspired observation space (optic flow).
The trained model is then analyzed with [SHAP](https://dl.acm.org/doi/10.5555/3295222.3295230) to obtain its attention patterns throughout an episode.

Preprint: [Understanding visual attention beehind bee-inspired UAV navigation](https://arxiv.org/abs/2507.11992)


Tested with Python 3.8 in Ubuntu 20.04

According to the [airsim installation instructions](https://microsoft.github.io/AirSim/build_linux/), this should also work on 18.04.
**DOES NOT WORK with Ubuntu 22.04**, as "vulkan-utils" is not a supported package in this distro. 

## installation

### installing this project

```
git clone https://github.com/pranavraj575/beehavior
pip3 install -e beehavior/.
pip3 install airsim
```

airsim installation is weird, as it requires numpy and msgpack-rpc-python to be installed already.
If installed simultaneously to either of these packages, it throws an error.

### installing/starting unreal engine

note: if you have access to the unreal project zip files, skip this until 'download zip and extract'

* create epic games account+github, link the two
* when linked, you will get an invite to the private epicgames developers
  repository (https://github.com/orgs/EpicGames/teams/developers)
* download the desired version **(USE 4.27!!)** of unreal engine from this private
  repo https://github.com/EpicGames/UnrealEngine
    * we used 4.27: https://github.com/EpicGames/UnrealEngine/tree/4.27
    * git clone did not work for me, can go to the branch, click <> code, then download zip and extract, (we put in /home, if you put it somwhere else, you may have to mess with setting directory locations)
* then run UnrealEngine/Setup.sh `$ cd UnrealEngine; bash Setup.sh`
* Generate project files:
  `$ bash GenerateProjectFiles.sh`
* then do make:
  `$ make`


### installing airsim

follow the [directions from Microsoft](https://microsoft.github.io/AirSim/build_linux/):

```
git clone https://github.com/Microsoft/AirSim.git
cd AirSim
./setup.sh
./build.sh
```

### installing our environment

Unfortunately our environment (the files for the tunnel and obstacles) are too large to store in this repository.
Obtain a copy of Blocks_4.27.tar.gz (contact me or email [a.veda@unsw.edu.au](mailto:a.veda@unsw.edu.au)), unzip, and place in the Airsim/Unreal/Enviornments folder.

_TODO_: Large file storage/linking?

## installation tests

### test installation of unreal engine with GUI:

* launch UE4E from the directory where you have UnrealEngine: `./Engine/Binaries/Linux/UE4Editor`
* Select the existing environment or create a new environment
* If selecting an existing environment, once it prompts you to convert the project before opening, select more options
  and click on skip conversion. This will then load the existing environment.

### test installation of unreal engine with command line:

* know the directory to the `UE4Editor` file, and the directory to an airsim `.uproject` file
* Run the project in game mode: `./<...>/Engine/Binaries/Linux/UE4Editor <...>/<PROJECT.uproject> -game -windowed`
* This should open the project in 'game' and 'windowed' mode, which does not load the Editor GUI, and allows you to
  click offscreen

### test basic keyboard input

* start a quadcopter project in game+windowed mode
  `<...>/Engine/Binaries/Linux/UE4Editor <...>/AirSim/Unreal/Environments/Blocks_4.27/Blocks.uproject -game -windowed`
* in another terminal, run `python3 airsim_interface/keyboard_test.py`
* control the drone!
    * keys 1234567890 control thrust, 1 is least and 0 is most
    * arrow keys control roll/pitch
    * space bar progresses simulation for a quarter second and pauses
    * c clears roll/pitch
    * r to reset simulation
    * i to view images
    * Q (shift + q) to stop python script

### test gym enviornment

* start a quadcopter project in game+windowed mode
  `<...>/Engine/Binaries/Linux/UE4Editor <...>/AirSim/Unreal/Environments/Blocks_4.27/Blocks.uproject -game -windowed`
* in another terminal, run `python3 beehavior/test/finger_gym.py`
* control the drone! (same controls as `airsim_interface/keyboard_test.py`)
