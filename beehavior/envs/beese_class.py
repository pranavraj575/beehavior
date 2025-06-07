from typing import Tuple, Optional

import airsim
import gymnasium as gym
from grpc.aio import AioRpcError
from gymnasium.core import ActType, ObsType
import numpy as np
import math
from collections import deque
from scipy.spatial.transform import Rotation

from airsim import Vector3r, Pose
from airsim_interface.interface import connect_client, disconnect_client, step, of_geo, get_of_geo_shape, get_depth_img


class BeeseClass(gym.Env):
    """
    Base class for controlling the quadrotor in unreal
    state is the current unreal engine state
    observation is a 2 vector of zeros right now
    Actions are roll [-1,1], pitch [-1,1], and thrust [0,1]
    reward is just -1 for collisions
    terminates upon collision
    """
    ACTION_ROLL_PITCH_THRUST = 'rpt'  # action bounds about np.pi/18 (10 degrees tilt)

    ACTION_VELOCITY = 'vel'  # action bounds about 1.5 m/s
    ACTION_VELOCITY_XY = 'vel_xy'  # action bounds aboutn 1.5 m/s

    ACTION_ACCELERATION = 'acc'  # action bounds about 3 m/s^2
    ACTION_ACCELERATION_XY = 'acc_xy'  # action bounds about 3 m/s^2

    def __init__(self,
                 client=None,
                 dt=.25,
                 action_bounds=None,
                 vehicle_name='',
                 real_time=False,
                 collision_grace=1,
                 initial_position=None,
                 action_type=ACTION_ACCELERATION,
                 velocity_bounds=2.,
                 fix_z_to=None,
                 timeout=300,
                 global_actions=False,
                 ):
        """
        Args:
            client: airsim interface client
                if None, makes own client
                if not a airsim.MultirotorClient object, makes a dummy environment that cannot use .step()
                    i.e. pass in client=False to do this
            dt: actions run for this amount of time
            action_bounds: maximum action magnitude that agent can take
                if action is ROLL_PITCH_YAW, this is max RADIANS that agent can roll/pitch
                if action is VELOCITY, this is maximum velocity (m/s)
                if action is ACCELERATION, this is maximum acceleration (m/s^2)
                    at each timestep, the acceleration*dt is added to the velocity
                if None, this will be the default bounds for each action type
            vehicle_name: name of vehicle
            real_time: if true, does not pause simulation after each step
            collision_grace: number of timesteps to forgive collisions after reset
            initial_position: initial position to go to after reset, box of ((x low, x high),(y low, y high), (z low, z high))
                each dim can be replaced by a value instead of a range
                also can be a dict of (initial pos box: prob), where the probs add to 1
            velocity_bounds:
                if acceleration ctrl, this is the maximum velocity magnitude on each dimension
            fix_z_to: if velocity_ctrl, fixes the height to a certin value, if None, doesnt do this
            timeout: seconds until env timeout
            global_actions: whether action vectors are in global frame
                NOTE: z vector is ALWAYS in global frame, x and y can be in global or local frame
                    i.e. command vector of (0,0,1) will always correspond to straight up, regardless of agent orientation
                        (1,0,0) is either positive x (global) or direction agent is facing (local)
        """
        super().__init__()

        if client is None:
            client = connect_client(client=client)
        self.client = client
        self.valid_client = (type(self.client) == airsim.MultirotorClient)
        if not self.valid_client:
            print("WARNING: dummy environment made without valid client, "
                  "will not be able to interface with simulator (cannot use step etc.)")

        self.vehicle_name = vehicle_name
        self.dt = dt
        self.action_bounds = action_bounds
        self.real_time = real_time
        self.collision_grace = collision_grace
        self.initial_pos = initial_position
        self.timeout = timeout
        self.env_time = 0  # count environment time in intervals of dt
        self.action_type = action_type
        self.fix_z_to = fix_z_to
        self.velocity_bounds = velocity_bounds
        self.global_actions = global_actions

        # in euclidean cases, we make the output on a [-1,1] box, then scale it to a radius action_bounds ball later
        # in roll, pitch, yaw, we do not scale as it doesnt really make (non euclidean)
        # we also keep the action space approximately a [-1,1] box, the network seems to make more sense if the
        #   output space is all around the same scale. we scale network outputs in the .step() function before sending
        #   the command to simulation
        if self.action_type == self.ACTION_VELOCITY:
            if self.action_bounds is None:
                self.action_bounds = 1.
            self.action_space = gym.spaces.Box(
                low=-1,
                high=1,
                shape=(3,),
                dtype=np.float64,
            )
        elif self.action_type == self.ACTION_VELOCITY_XY:
            if self.action_bounds is None:
                self.action_bounds = 1.
            self.action_space = gym.spaces.Box(
                low=-1,
                high=1,
                shape=(2,),
                dtype=np.float64,
            )
        elif self.action_type == self.ACTION_ACCELERATION:
            if self.action_bounds is None:
                self.action_bounds = 3.
            self.velocity_target = np.zeros(3)
            self.action_space = gym.spaces.Box(
                low=-1,
                high=1,
                shape=(3,),
                dtype=np.float64,
            )
        elif self.action_type == self.ACTION_ACCELERATION_XY:
            if self.action_bounds is None:
                self.action_bounds = 3.
            self.velocity_target = np.zeros(2)
            self.action_space = gym.spaces.Box(
                low=-1,
                high=1,
                shape=(2,),
                dtype=np.float64,
            )
        elif self.action_type == self.ACTION_ROLL_PITCH_THRUST:
            if self.action_bounds is None:
                self.action_bounds = np.pi/36  # 5 degrees, which apparently is quite steep
            # in .step(), roll and pitch will be multiplied by self.action_bounds
            self.action_space = gym.spaces.Box(
                low=np.array([-1, -1, 0]),
                high=np.array([1, 1, 1]),
                shape=(3,),
                dtype=np.float64,
            )
        else:
            raise NotImplementedError
        self.observation_space = self.define_observation_space()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        """
        implementiaon of gym's step function
        Args:
            action: action that agent makes at a timestep
        Returns:
            (observation, reward, termination,
                    truncation, info)
        """
        pose = self.get_pose()
        action = action.copy()  # to prevent mutation of action
        # orient_eulerian = self.get_orientation_eulerian(
        #    quaternion=(pose.orientation.x_val, pose.orientation.y_val, pose.orientation.z_val, pose.orientation.w_val)
        # )
        if self.action_type == self.ACTION_VELOCITY:
            vec = self.map_vec_to_ball(vector=action,
                                       radius=self.action_bounds,
                                       idxs=None,
                                       )
            if not self.global_actions:
                vec[:2] = self.to_global_vec(vec=vec[:2], pose=pose)
            vx, vy, vz = vec
            cmd = lambda: self.client.moveByVelocityAsync(vx=vx,
                                                          vy=vy,
                                                          vz=vz,
                                                          duration=self.dt,
                                                          vehicle_name=self.vehicle_name,
                                                          )
        elif self.action_type == self.ACTION_VELOCITY_XY:
            target_ht = self.fix_z_to if self.fix_z_to is not None else pose.position.z_val
            vec = self.map_vec_to_ball(vector=action,
                                       radius=self.action_bounds,
                                       idxs=None,
                                       )
            if not self.global_actions:
                vec = self.to_global_vec(vec=vec, pose=pose)
            vx, vy = vec
            cmd = lambda: self.client.moveByVelocityZAsync(vx=vx,
                                                           vy=vy,
                                                           z=target_ht,
                                                           duration=self.dt,
                                                           vehicle_name=self.vehicle_name,
                                                           )
        elif self.action_type == self.ACTION_ACCELERATION:
            acc = self.map_vec_to_ball(vector=action,
                                       radius=self.action_bounds,
                                       idxs=None,
                                       )
            self.velocity_target = self.velocity_target + acc*self.dt
            speed = np.linalg.norm(self.velocity_target)
            if speed > self.velocity_bounds:
                self.velocity_target = self.velocity_target*(self.velocity_bounds/speed)
            vec = self.velocity_target
            if not self.global_actions:
                vec[:2] = self.to_global_vec(vec=vec[:2], pose=pose)
            vx, vy, vz = vec
            cmd = lambda: self.client.moveByVelocityAsync(vx=vx,
                                                          vy=vy,
                                                          vz=vz,
                                                          duration=self.dt,
                                                          vehicle_name=self.vehicle_name,
                                                          )
        elif self.action_type == self.ACTION_ACCELERATION_XY:
            acc = self.map_vec_to_ball(vector=action,
                                       radius=self.action_bounds,
                                       idxs=None,
                                       )
            target_ht = self.fix_z_to if self.fix_z_to is not None else pose.position.z_val
            self.velocity_target = self.velocity_target + acc*self.dt
            speed = np.linalg.norm(self.velocity_target)
            if speed > self.velocity_bounds:
                self.velocity_target = self.velocity_target*(self.velocity_bounds/speed)
            vec = self.velocity_target
            if not self.global_actions:
                vec = self.to_global_vec(vec=vec, pose=pose)

            vx, vy = vec
            cmd = lambda: self.client.moveByVelocityZAsync(vx=vx,
                                                           vy=vy,
                                                           z=target_ht,
                                                           duration=self.dt,
                                                           vehicle_name=self.vehicle_name,
                                                           )
        elif self.action_type == self.ACTION_ROLL_PITCH_THRUST:
            roll, pitch, thrust = action
            roll, pitch = roll*self.action_bounds, pitch*self.action_bounds
            cmd = lambda: self.client.moveByRollPitchYawrateThrottleAsync(roll=roll,
                                                                          pitch=pitch,
                                                                          yaw_rate=0,
                                                                          throttle=thrust,
                                                                          duration=self.dt,
                                                                          vehicle_name=self.vehicle_name,
                                                                          )
        else:
            raise NotImplementedError
        step(client=self.client,
             seconds=self.dt,
             cmd=cmd,
             pause_after=not self.real_time,
             )

        if self.col_cnt > 0:
            # if we are still in the takeoff grace period, just update so ground is not registered as a collision
            self.update_recent_colision()

        collided = self.has_collided()
        self.env_time += self.dt  # TODO: maybe query client or something? this is prob sufficient
        obs = self.get_obs()

        r = self.get_rwd(collided=collided, obs=obs)

        self.col_cnt = max(0, self.col_cnt - 1)
        term, trunc = self.get_termination(collided)

        info = {
            'collided': collided,
        }
        if type(r) == tuple:
            r, temp_dic = r
            info.update(temp_dic)
        # observation, reward, termination, truncation, info
        return obs, r, term, trunc, info

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        """
        implementiaon of gym's reset function
        Args:
            seed: random seed
            options: option dictionary
        Returns:
            (observation, info dict)
        """
        super().reset(seed=seed)
        if self.valid_client:
            self.client.reset()
            self.client = connect_client(client=self.client)
        self.update_recent_colision()
        self.col_cnt = self.collision_grace
        self.env_time = 0
        initial_pos = None
        if self.initial_pos is not None:
            initial_pos = self.initial_pos
        if (options is not None) and ('initial_pos' in options) and (options['initial_pos'] is not None):
            # overrides default
            initial_pos = options['initial_pos']

        if self.valid_client and initial_pos is not None:
            if type(initial_pos) == dict:  # if initial_pos is a probability dict of boxes, choose which one to use
                r = np.random.rand()
                temp = None
                for (temp, rp) in initial_pos.items():
                    r -= rp
                    if r < 0:
                        break
                initial_pos = temp

            initial_pos = [d[0] + np.random.rand()*(d[1] - d[0]) if type(d) == tuple else d
                           for d in initial_pos
                           ]
            x, y, z = initial_pos

            position = Vector3r(x, y, z)
            # heading = AirSimClientBase.toQuaternion(roll, pitch, yaw)
            heading = None
            pose = Pose(position, heading)
            self.client.simSetVehiclePose(pose, ignore_collision=True, vehicle_name=self.vehicle_name)

            # have a non-zero initial velocity to prevent weird teleportation errors
            for _ in range(2):
                step(self.client,
                     seconds=None,
                     cmd=lambda: self.client.moveByVelocityAsync(np.random.rand() - .5,
                                                                 np.random.rand() - .5,
                                                                 np.random.rand() - .5,
                                                                 duration=self.dt,
                                                                 ).join(),
                     pause_after=not self.real_time,
                     )
            step(self.client,
                 seconds=None,
                 cmd=lambda: self.client.moveToPositionAsync(x, y, z, 1, ).join(),
                 pause_after=not self.real_time,
                 )
        if self.action_type in (self.ACTION_ACCELERATION, self.ACTION_ACCELERATION_XY):
            self.velocity_target = np.zeros_like(self.velocity_target)
        # obs, info
        return self.get_obs(), {}

    def close(self):
        """
        close client by disconnecting
        """
        super().close()
        disconnect_client(client=self.client)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # COLLISION STUFF
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def has_collided(self):
        """
        Returns: if agent has collided with anything new since last reset
        """
        return self.last_collision != self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name).time_stamp

    def update_recent_colision(self):
        """
        ignore the last thing agent collided with
            useful since client.reset() does not reset collisions
            also collisions with ground before takeoff counts
        """
        self.last_collision = self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name).time_stamp

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # TERMINATION STUFF
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def get_termination(self, collided):
        """
        returns whether to terminate/truncate an episode based on whether agent has collided
        Args:
            collided: whether agent collided with an obstacle
        Returns:
            terminate episode, truncate episode
        """
        termination = collided or self.env_time > self.timeout
        return termination, False

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # OBSERVATION STUFF
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def get_obs_vector(self):
        """
        gets observation vector from last completed step()
        """
        return np.arange(2)*1.0

    def get_obs_vector_dim(self):
        """
        dimension of self.get_obs_vector()
        """
        return 2

    def get_obs(self):
        """
        returns observation from last step()
        in the base class, returns a simple vector with no information
        """
        return self.get_obs_vector()

    def get_obs_shape(self):
        """
        returns shape of observation
        """
        return (self.get_obs_vector_dim(),)

    def define_observation_space(self):
        """
        to be run in __init__, defines the observation space of the environment
        """
        # REDEFINE THIS, currently assumes self.get_obs_vector() is between 0 and inf
        shape = self.get_obs_shape()
        return gym.spaces.Box(low=0, high=np.inf, shape=shape, dtype=np.float64)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # RWD STUFF
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def get_rwd(self, collided, obs):
        """
        gets reward from last completed step() call
        Args:
            collided: wthr the agent has collided with a wall (no need to recalculate for both termination and reward)
            obs: observation
        Returns:
            either a float, or a (float, dic) pair where dic is used to update info_dic
        """
        return -float(collided), dict()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # UTIL STUFF
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    def get_pose(self):
        """
        gets pose of agent in environment, pose object has a .position and .orientation
        """
        return self.client.simGetVehiclePose(vehicle_name=self.vehicle_name)

    def get_kinematics(self):
        """
        gets pose of agent in environment, pose object has a .position and .orientation
        """
        return self.client.simGetGroundTruthKinematics(vehicle_name=self.vehicle_name)

    def _get_rotation(self, pose=None):
        """
        gets scipy Rotation object from pose (if given) or self.get_pose()
        """
        if pose is None:
            pose = self.get_pose()
        o = pose.orientation

        quaternion = (o.x_val, o.y_val, o.z_val, o.w_val)
        return Rotation.from_quat(quaternion)  # scalar (w) is ordered last, the default

    def get_orientation_eulerian(self, pose=None):
        """
        returns roll,pitch,yaw
        Args:
            pose: if specified pose uses this
                else, uses self.get_pose()
        """
        r = self._get_rotation(pose=pose)
        return r.as_euler('xyz')

    def to_global_vec(self, vec, pose=None):
        """
        rotates vector to global coordinates
        i.e. a vector of (1,0) (positive x) will be rotated to the vector in the 'forward' drone direction
        Args:
            vec: 2d xy vector to rotate OR 3d xyz vector to rotate
        """
        r = self._get_rotation(pose=pose)
        R = r.as_matrix()
        if len(vec) == 2:
            return R[:2, :2]@vec
        elif len(vec) == 3:
            return R@vec
        else:
            raise NotImplementedError

    def map_vec_to_ball(self, vector, radius=1., idxs=None, ):
        """
        assumes a vector (np array) is in a [-1,1] box on certian indices
        rescales these indices to a ball of specified radius
        i.e. vector (1,-1) in 2d becomes (1/sqrt(2),-1/sqrt(2)) for radius 1
            vector (1/2,-1/2) becomes (1/2sqrt(2),-1/2sqrt(2))
        does this by linearly squishing each possible direction
        Args:
            vector: 1 dim np array to be rescaled, assumed each relevant index is within [-1,1]
            radius: radius of ball to rescale to
            idxs: list of indices to consider, if None, considers all
        """
        # TODO: REMOVE THESE LINES ONCE DONE TESTING
        #  map to a [-radius, radius] box for testing
        vector[idxs] = vector[idxs]*radius
        return vector

        if idxs is None:
            idxs = list(range(len(vector)))
        mag = np.linalg.norm(vector[idxs])
        if mag == 0:
            # vector is zero on these dims
            return vector
        united = vector[idxs]/mag
        scale = np.max(np.abs(united))
        # unit vector/scale will be on boundary of [-1,1] box
        # then original vector * scale will be within unit circle
        vector[idxs] = vector[idxs]*scale*radius
        return vector


class OFBeeseClass(BeeseClass):
    """
    dict observation, optic flow cameras ('front', 'bottom', etc) and 'vec' to vector
        OF observation is specifically magnitude of optic flow of points projected onto sphere around observer
    """
    # these are used to specify what the agent can sense as an image
    INPUT_RAW_OF = 'INPUT_RAW_OF'
    # geometric optic flow
    INPUT_LOG_OF = 'INPUT_LOG_OF'
    # geometric optic flow, scaled by log
    INPUT_OF_ORIENTATION = 'INPUT_OF_ORIENTATION'
    #  whether bee can see the orientation of OF
    INPUT_INV_DEPTH_IMG = 'INPUT_INV_DEPTH_IMG'

    # give agent 1/depth image
    #   used to confirm whether a learning task is possible with depth information

    def __init__(self,
                 of_cameras=('front', 'bottom'),
                 default_camera_shape=(2, 240, 320),
                 img_history_steps=2,
                 input_img_space=(INPUT_LOG_OF, INPUT_OF_ORIENTATION,),
                 of_ignore_angular_velocity=True,
                 concatenate_observations=False,
                 **kwargs,
                 ):
        """
        Args:
            client:
            dt:
            action_bounds:
            vehicle_name:
            real_time:
            of_cameras: camera to take OF information from, if tuple, input space is a dict of images
                this will be acted on independently by the cnns, then the output vectors will be joined
            default_camera_shape: default shape for cameras, used when client is not defined
            img_history_steps: number of images to show at each time step
            of_mapping: mapping to apply to optic flow
            input_img_space: list of keys that determines what the agent can visually see, keys are
                RAW_OF, LOG_OF, OF_ORIENTATION, INV_DEPTH_IMG
            see_of_orientation: whether bee can see the orientation of OF
            of_ignore_angular_velocity: whether to ignore angular velocity in OF calc
                if true, pretends camera is on chicken head
            concatenate_observations: instead of dict observation space, concatenates everything into a long row vector
                used to prevent issues with SHAP package
        """
        self.concat_obs=concatenate_observations

        self.obs_shape = None
        self.img_stack = None
        self.of_ignore_angular_velocity = of_ignore_angular_velocity
        self.of_cameras = of_cameras
        self.default_camera_shape = default_camera_shape
        self.input_img_space = set(input_img_space)
        self.ordered_input_img_space = tuple([input_key for input_key in (self.INPUT_RAW_OF,
                                                                          self.INPUT_LOG_OF,
                                                                          self.INPUT_OF_ORIENTATION,
                                                                          self.INPUT_INV_DEPTH_IMG,
                                                                          )
                                              if input_key in self.input_img_space])

        self.imgs_per_step = (int(self.INPUT_RAW_OF in self.input_img_space) +
                              int(self.INPUT_LOG_OF in self.input_img_space) +
                              2*int(self.INPUT_OF_ORIENTATION in self.input_img_space) +
                              int(self.INPUT_INV_DEPTH_IMG in self.input_img_space)
                              )
        self.img_stack_size = img_history_steps*self.imgs_per_step
        super().__init__(
            **kwargs,
        )

    def define_observation_space(self):
        """
        defines gym observation space, taking optic flow image and appending a vector to each element
        """
        C = self.img_stack_size  # stack this many images on top of each other
        shape = self.get_obs_shape()
        obs_space_dic = dict()
        for k, (_, H, W) in zip(self.of_cameras, self.get_of_data_shape()):
            sh = shape[k]
            arr = np.ones(sh)
            # (scaled) magnitudes are from -inf to inf
            low = -np.inf*arr
            high = np.inf*arr
            i = 0
            if self.INPUT_RAW_OF in self.input_img_space:
                # sees (...,raw_OF,...) at each timestep
                # OF is on [0,inf)
                low[i:C:self.imgs_per_step, :, :] = 0
                i += 1
            if self.INPUT_LOG_OF in self.input_img_space:
                # sees (log(OF),...) at each timestep
                # log(OF) is on (-inf,inf) (we clip to avoid log(0) error)
                i += 1
            if self.INPUT_OF_ORIENTATION in self.input_img_space:
                # sees (..., scaled x component, scaled y component,...) at each timestep
                # components are -1 to 1
                for dim in range(2):
                    low[i:C:self.imgs_per_step, :, :] = -1
                    high[i:C:self.imgs_per_step, :, :] = 1
                    i += 1
            if self.INPUT_INV_DEPTH_IMG in self.input_img_space:
                # sees (...,depth_img,...) at each timestep
                # depth image and inv depth img is [0,inf)
                low[i:C:self.imgs_per_step, :, :] = 0
                high[i:C:self.imgs_per_step, :, :] = np.inf
                i += 1
            low[C:, :, :] = -np.inf
            high[C:, :, :] = np.inf
            obs_space_dic[k] = gym.spaces.Box(low=low, high=high, shape=sh, dtype=np.float64)
        if self.get_obs_vector_dim() > 0:
            obs_space_dic['vec'] = gym.spaces.Box(low=-np.inf,
                                                  high=np.inf,
                                                  shape=(self.get_obs_vector_dim(),),
                                                  dtype=np.float64,
                                                  )
        self.ordered_keys=tuple(sorted(list(obs_space_dic.keys())))
        if self.concat_obs:
            raise NotImplementedError
        else:
            return gym.spaces.Dict(obs_space_dic)

    def get_obs(self):
        obs = dict()
        for camera_name in self.of_cameras:
            of = of_geo(client=self.client,
                        camera_name=camera_name,
                        vehicle_name=self.vehicle_name,
                        ignore_angular_velocity=self.of_ignore_angular_velocity,
                        )
            of_magnitude = np.linalg.norm(of, axis=0)  # magnitude of x and y components of projected optic flow
            obs[camera_name] = None
            # H, W = of.shape
            if self.INPUT_RAW_OF in self.input_img_space:
                # (H,W) optic flow magnitude  on [0,inf)
                self.img_stack[camera_name].append(of_magnitude.copy())
            if self.INPUT_LOG_OF in self.input_img_space:
                # (H,W) log(optic flow magnitude)  on (-inf,inf)
                # clipped to avoid log(0) error
                self.img_stack[camera_name].append(np.log(np.clip(of_magnitude, 10e-3, np.inf)))
            if self.INPUT_OF_ORIENTATION in self.input_img_space:
                # 2x (H,W) for x and y components of optic flow orientation
                #  each component is -1 to 1
                clipped_mag = np.clip(of_magnitude, 10e-4, np.inf)  # avoid division by zero
                self.img_stack[camera_name].append(of[0]/clipped_mag)
                self.img_stack[camera_name].append(of[1]/clipped_mag)
            if self.INPUT_INV_DEPTH_IMG in self.input_img_space:
                # sees (...,inv_depth_img,...) at each timestep
                # depth image and inv depth img are on (0,inf)
                depth = get_depth_img(client=self.client,
                                      camera_name=camera_name,
                                      numpee=True,
                                      )

                # clip depth to avoid 1/0 error, this means minimum visible depth is .001m which is resonable
                self.img_stack[camera_name].append(1/np.clip(depth, 10e-3, np.inf))

            while len(self.img_stack[camera_name]) < self.img_stack_size:
                # copy the first however many elements
                extensor = [self.img_stack[camera_name][i] for i in range(self.imgs_per_step)]
                self.img_stack[camera_name].extend(extensor)
            obs[camera_name] = np.stack(self.img_stack[camera_name],
                                        axis=0)  # this is a (C,H,W) history of optic flow data

        if self.get_obs_vector_dim() > 0:
            obs['vec'] = self.get_obs_vector()

        return obs

    def get_obs_shape(self):
        if self.obs_shape is None:
            self.obs_shape = dict()
            of_shape = self.get_of_data_shape()
            for k, of_sh in zip(self.of_cameras, of_shape):
                # self.obs_shape = (of_shape[0] + self.get_obs_vector_dim(), *of_shape[1:])
                # we are only using translational optic flow (1,H,W), and stacking self.img_stack_size of them
                self.obs_shape[k] = (self.img_stack_size, *of_sh[1:])
            self.obs_shape['vec'] = (self.get_obs_vector_dim(),)
        return self.obs_shape

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        self.img_stack = {k: deque(maxlen=self.img_stack_size) for k in self.of_cameras}
        stuf = super().reset(seed=seed, options=options)
        return stuf

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # OF STUFF
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def get_of_data_shape(self):
        """
        shape of self.get_of_data()
        costly, should not be run too many times, as we can either save this shape or just look at the last observation
        """
        if self.valid_client:
            shape = get_of_geo_shape(client=self.client, camera_name=self.of_cameras)
        else:
            shape = tuple(self.default_camera_shape for cam_name in self.of_cameras)
        return shape


if __name__ == '__main__':
    import time

    env=OFBeeseClass(of_cameras=('front', 'bottom'),
                     concatenate_observations=False,
                     client=False,
                     )
    print(type(env.observation_space.sample()))
    env=OFBeeseClass(of_cameras=('front', 'bottom'),
                     concatenate_observations=True,
                     client=False,
                     )
    print(type(env.observation_space.sample()))
    quit()

    env = OFBeeseClass(dt=.2, real_time=False, action_type=OFBeeseClass.ACTION_VELOCITY_XY,
                       of_cameras=('front', 'bottom'))
    env.reset()
    env.step(action=env.action_space.sample())
    for _ in range(0, int(1/env.dt), 1):
        env.step(action=env.action_space.sample())
    for i in range(0, int(10/env.dt), 1):
        action = env.action_space.sample()
        print(action)
        obs, rwd, term, _, _ = env.step(action=action)
        if term:
            print("CRASHED")
            break
    env.close()

    env = BeeseClass(dt=.2, real_time=True)
    env.reset()
    env.step(action=np.array([0., 0., 1.]))
    for _ in range(0, int(1/env.dt), 1):
        env.step(action=np.array([0., 0., 1.]))
    for i in range(0, int(10/env.dt), 1):
        action = env.action_space.sample()
        r, p, t = action
        obs, rwd, term, _, _ = env.step(action=action)
        print('roll:', r, 'pitch:', p, 'thrust:', t, 'reward:', rwd)
        if term:
            print("CRASHED")
            break
    env.close()
