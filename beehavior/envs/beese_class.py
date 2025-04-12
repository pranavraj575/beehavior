from typing import Tuple, Optional
import gymnasium as gym
from gymnasium.core import ActType, ObsType
import numpy as np
import math
from collections import deque

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

    def __init__(self,
                 client=None,
                 dt=.25,
                 max_tilt=np.pi/18,
                 vehicle_name='',
                 of_camera='front',
                 real_time=False,
                 collision_grace=1,
                 initial_position=None,
                 velocity_ctrl=False,
                 fix_z_to=None,
                 timeout=300,
                 ):
        """
        Args:
            client: airsim interface client
                if None, makes own client
            dt: actions run for this amount of time
            max_tilt: maximum RADIANS that agent can roll/pitch
            vehicle_name: name of vehicle
            real_time: if true, does not pause simulation after each step
            collision_grace: number of timesteps to forgive collisions after reset
            initial_position: initial position to go to after reset, box of ((x low, x high),(y low, y high), (z low, z high))
                each dim can be replaced by a value instead of a range
                also can be a dict of (initial pos box: prob), where the probs add to 1
            velocity_ctrl: output velocity command (x,y,z) instead of r,p,thrust
            fix_z_to: if velocity_ctrl, fixes the height to a certin value, if None, doesnt do this
            timeout: seconds until env timeout
        """
        super().__init__()
        if client is None:
            client = connect_client(client=client)
        self.client = client

        self.vehicle_name = vehicle_name
        self.of_camera = of_camera
        self.dt = dt
        self.max_tilt = max_tilt
        self.real_time = real_time
        self.collision_grace = collision_grace
        self.initial_pos = initial_position
        self.timeout = timeout
        self.env_time = 0  # count environment time in intervals of dt
        self.velocity_ctrl = velocity_ctrl
        self.fix_z_to = fix_z_to

        if self.velocity_ctrl:
            self.action_space = gym.spaces.Box(
                low=np.array([-1, -1, -1]),
                high=np.array([1, 1, 1]),
                shape=(3,),
                dtype=np.float64,
            )
        else:
            self.action_space = gym.spaces.Box(
                low=np.array([-self.max_tilt, -self.max_tilt, 0]),
                high=np.array([self.max_tilt, self.max_tilt, 1]),
                shape=(3,),
                dtype=np.float64,
            )
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
        if self.velocity_ctrl:
            vx, vy, vz = action
            if self.fix_z_to is not None:
                cmd = lambda: self.client.moveByVelocityZAsync(vx=vx,
                                                               vy=vy,
                                                               z=self.fix_z_to,
                                                               duration=self.dt,
                                                               vehicle_name=self.vehicle_name,
                                                               )
            else:
                cmd = lambda: self.client.moveByVelocityAsync(vx=vx,
                                                              vy=vy,
                                                              vz=vz,
                                                              duration=self.dt,
                                                              vehicle_name=self.vehicle_name,
                                                              )

        else:
            roll, pitch, thrust = action
            cmd = lambda: self.client.moveByRollPitchYawrateThrottleAsync(roll=roll,
                                                                          pitch=pitch,
                                                                          yaw_rate=0,
                                                                          throttle=thrust,
                                                                          duration=self.dt,
                                                                          vehicle_name=self.vehicle_name,
                                                                          )
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
        self.client.reset()
        self.client = connect_client(client=self.client)
        self.update_recent_colision()
        self.col_cnt = self.collision_grace
        self.env_time = 0
        initial_pos = None
        if self.initial_pos is not None:
            initial_pos = self.initial_pos
        if options is not None and 'initial_pos' in options:
            # overrides default
            initial_pos = options['initial_pos']

        if initial_pos is not None:
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

    def get_pose(self):
        """
        gets pose of agent in environment, pose object has a .position and .orientation
        """
        return self.client.simGetVehiclePose(vehicle_name=self.vehicle_name)

    def get_orientation_eulerian(self, quaternion=None):
        """
        returns roll,pitch,yaw
        Args:
            quaternion: if specified (x,y,z,w) uses this
                else, uses self.get_pose()
        """
        if quaternion is None:
            o = self.get_pose().orientation
            quaternion = (o.x_val, o.y_val, o.z_val, o.w_val)
        x, y, z, w = quaternion

        # convert to rpy
        t0 = +2.0*(w*x + y*z)
        t1 = +1.0 - 2.0*(x*x + y*y)
        rl = math.atan2(t0, t1)

        t2 = +2.0*(w*y - z*x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        ptch = math.asin(t2)

        t3 = +2.0*(w*z + x*y)
        t4 = +1.0 - 2.0*(y*y + z*z)
        yw = math.atan2(t3, t4)

        return rl, ptch, yw

    def get_of_data(self):
        """
        gets np array of optic flow from last completed step()
        """
        return of_geo(client=self.client, camera_name=self.of_camera, vehicle_name=self.vehicle_name)

    def get_of_data_shape(self):
        """
        shape of self.get_of_data()
        costly, should not be run too many times, as we can either save this shape or just look at the last observation
        """
        return get_of_geo_shape(client=self.client, camera_name=self.of_camera)

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


class OFBeeseClass(BeeseClass):
    """
    observation is optic flow data, with a vector (self.get_obs_vector()) appended to every pixel
        observation is specifically magnitude of optic flow of points projected onto sphere around observer
    the reason we do it this way is because gym.space.Tuple is not supported by stable_baselines3
        we could make our own networks, but just appending is easier
    """
    # these are used to specify what the agent can sense as an image
    RAW_OF = 'RAW_OF'
    # geometric optic flow
    LOG_OF = 'LOG_OF'
    # geometric optic flow, scaled by log
    OF_ORIENTATION = 'OF_ORIENTATION'
    #  whether bee can see the orientation of OF
    INV_DEPTH_IMG = 'INV_DEPTH_IMG'

    # give agent 1/depth image
    #   used to confirm whether a learning task is possible with depth information

    def __init__(self,
                 client=None,
                 dt=.25,
                 max_tilt=np.pi/18,
                 vehicle_name='',
                 real_time=False,
                 collision_grace=1,
                 initial_position=None,
                 timeout=300,
                 img_history_steps=2,
                 input_img_space=(LOG_OF, OF_ORIENTATION,),
                 velocity_ctrl=False,
                 fix_z_to=None,
                 of_ignore_angular_velocity=True,
                 ):
        """
        Args:
            client:
            dt:
            max_tilt:
            vehicle_name:
            real_time:
            collision_grace:
            initial_position:
            timeout:
            img_history_steps: number of images to show at each time step
            of_mapping: mapping to apply to optic flow
            input_img_space: list of keys that determines what the agent can visually see, keys are
                RAW_OF, LOG_OF, OF_ORIENTATION, INV_DEPTH_IMG
            see_of_orientation: whether bee can see the orientation of OF
            of_ignore_angular_velocity: whether to ignore angular velocity in OF calc
                if true, pretends camera is on chicken head
        """
        self.obs_shape = None
        self.img_stack = None
        self.of_ignore_angular_velocity = of_ignore_angular_velocity
        self.input_img_space = set(input_img_space)
        self.imgs_per_step = (int(self.RAW_OF in self.input_img_space) +
                              int(self.LOG_OF in self.input_img_space) +
                              2*int(self.OF_ORIENTATION in self.input_img_space) +
                              int(self.INV_DEPTH_IMG in self.input_img_space)
                              )
        self.img_stack_size = img_history_steps*self.imgs_per_step
        super().__init__(
            client=client,
            dt=dt,
            max_tilt=max_tilt,
            vehicle_name=vehicle_name,
            real_time=real_time,
            collision_grace=collision_grace,
            initial_position=initial_position,
            timeout=timeout,
            velocity_ctrl=velocity_ctrl,
            fix_z_to=fix_z_to,
        )

    def define_observation_space(self):
        """
        defines gym observation space, taking optic flow image and appending a vector to each element
        """
        (_, H, W) = self.get_of_data_shape()
        C = self.img_stack_size  # stack this many images on top of each other
        shape = self.get_obs_shape()
        arr = np.ones(shape)

        # (scaled) magnitudes are from -inf to inf
        low = -np.inf*arr
        high = np.inf*arr
        i = 0
        if self.RAW_OF in self.input_img_space:
            # sees (...,raw_OF,...) at each timestep
            # OF is on [0,inf)
            low[i:C:self.imgs_per_step, :, :] = 0
            i += 1
        if self.LOG_OF in self.input_img_space:
            # sees (log(OF),...) at each timestep
            # log(OF) is on (-inf,inf) (we clip to avoid log(0) error)
            i += 1
        if self.OF_ORIENTATION in self.input_img_space:
            # sees (..., scaled x component, scaled y component,...) at each timestep
            # components are -1 to 1
            for dim in range(2):
                low[i:C:self.imgs_per_step, :, :] = -1
                high[i:C:self.imgs_per_step, :, :] = 1
                i += 1
        if self.INV_DEPTH_IMG in self.input_img_space:
            # sees (...,depth_img,...) at each timestep
            # depth image and inv depth img is [0,inf)
            low[i:C:self.imgs_per_step, :, :] = 0
            high[i:C:self.imgs_per_step, :, :] = np.inf
            i += 1
        low[C:, :, :] = -np.inf
        high[C:, :, :] = np.inf
        return gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.float64)

    def get_obs(self):
        of = of_geo(client=self.client,
                    camera_name='front',
                    vehicle_name=self.vehicle_name,
                    ignore_angular_velocity=self.of_ignore_angular_velocity,
                    )
        of_magnitude = np.linalg.norm(of, axis=0)  # magnitude of x and y components of projected optic flow
        # H, W = of.shape
        if self.RAW_OF in self.input_img_space:
            # (H,W) optic flow magnitude  on [0,inf)
            self.img_stack.append(of_magnitude.copy())
        if self.LOG_OF in self.input_img_space:
            # (H,W) log(optic flow magnitude)  on (-inf,inf)
            # clipped to avoid log(0) error
            self.img_stack.append(np.log(np.clip(of_magnitude, 10e-3, np.inf)))
        if self.OF_ORIENTATION in self.input_img_space:
            # 2x (H,W) for x and y components of optic flow orientation
            #  each component is -1 to 1
            clipped_mag = np.clip(of_magnitude, 10e-4, np.inf)  # avoid division by zero
            self.img_stack.append(of[0]/clipped_mag)
            self.img_stack.append(of[1]/clipped_mag)
        if self.INV_DEPTH_IMG in self.input_img_space:
            # sees (...,inv_depth_img,...) at each timestep
            # depth image and inv depth img are on (0,inf)
            depth = get_depth_img(client=self.client,
                                  camera_name='front',
                                  numpee=True,
                                  )
            # clip depth to avoid 1/0 error, this means minimum visible depth is .001m which is resonable
            self.img_stack.append(1/np.clip(depth, 10e-3, np.inf))

        while len(self.img_stack) < self.img_stack_size:
            # copy the first however many elements
            extensor = [self.img_stack[i] for i in range(self.imgs_per_step)]
            self.img_stack.extend(extensor)

        vec = self.get_obs_vector()
        # (m,) sized vector, goal conditioning
        obs = np.zeros(self.get_obs_shape())
        # (C+m,H,W)
        C = self.img_stack_size  # number of of images
        obs[:C, :, :] = np.stack(self.img_stack, axis=0)  # this is a (C,H,W) history of optic flow data
        obs[C:, :, :] = np.expand_dims(vec, axis=(1, 2))  # (m,1,1)

        # places copies of vec at every pixel
        # can technically use gym.spaces.Tuple, and return (of, vec)
        #  unfortunately Tuple space is not supported in stable_baselines
        return obs

    def get_obs_shape(self):
        if self.obs_shape is None:
            of_shape = self.get_of_data_shape()
            # self.obs_shape = (of_shape[0] + self.get_obs_vector_dim(), *of_shape[1:])
            # we are only using translational optic flow (1,H,W), and stacking self.img_stack_size of them
            self.obs_shape = (self.img_stack_size + self.get_obs_vector_dim(), *of_shape[1:])
        return self.obs_shape

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        self.img_stack = deque(maxlen=self.img_stack_size)
        stuf = super().reset(seed=seed, options=options)
        return stuf


if __name__ == '__main__':
    import time

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
