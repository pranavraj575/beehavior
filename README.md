# beehavior

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

* create epic games account+github, link the two
* when linked, you will get an invite to the private epicgames developers
  repository (https://github.com/orgs/EpicGames/teams/developers)
* download the desired version of unreal engine from this private repo https://github.com/EpicGames/UnrealEngine
    * we used 4.27: https://github.com/EpicGames/UnrealEngine/tree/4.27
    * git clone did not work for me, can go to the branch, click <> code, then download zip and extract,
* then run UnrealEngine/Setup.sh `$ cd UnrealEngine; bash Setup.sh`
* Generate project files:
  `$ bash GenerateProjectFiles.sh`
* then do make:
  `$ make`

#### Test installation with GUI:

* launch UE4E from the directory where you have UnrealEngine: `./Engine/Binaries/Linux/UE4Editor`
* Select the existing environment or create a new environment
* If selecting an existing environment, once it prompts you to convert the project before opening, select more options
  and click on skip conversion. This will then load the existing environment.

#### Test installation with command line:

* know the directory to the `UE4Editor` file, and the directory to an airsim `.uproject` file
* Run the project in game mode: `./<...>/Engine/Binaries/Linux/UE4Editor <...>/<PROJECT.uproject> -game`
* This should open the project in 'game' mode, which does not load the Editor GUI

