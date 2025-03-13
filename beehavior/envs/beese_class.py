from typing import Tuple, Optional
import gymnasium as gym
from gymnasium.core import ActType, ObsType
import numpy as np
import math

from airsim_interface.interface import connect_client, disconnect_client, step, of_geo, get_of_geo_shape


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
            initial_position: initial position to go to after reset
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

        self.action_space = gym.spaces.Box(low=np.array([-1, -1, 0]), high=1, shape=(3,), dtype=np.float64)
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
        roll, pitch, thrust = action
        step(client=self.client,
             seconds=self.dt,
             cmd=lambda: self.client.moveByRollPitchYawrateThrottleAsync(roll=roll,
                                                                         pitch=pitch,
                                                                         yaw_rate=0,
                                                                         throttle=thrust,
                                                                         duration=self.dt,
                                                                         vehicle_name=self.vehicle_name,
                                                                         ),
             pause_after=not self.real_time,
             )

        if self.col_cnt > 0:
            # if we are still in the takeoff grace period, just update so ground is not registered as a collision
            self.update_recent_colision()

        collided = self.has_collided()
        obs = self.get_obs()

        r = self.get_rwd(collided=collided, obs=obs)

        self.col_cnt = max(0, self.col_cnt - 1)
        term, trunc = self.get_termination(collided)

        # observation, reward, termination, truncation, info
        return obs, r, term, trunc, {}

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
        if self.initial_pos is not None:
            x, y, z = self.initial_pos
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
        return collided, False

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
        return of_geo(client=self.client, camera_name=self.of_camera, vehicle_name=self.vehicle_name, FOVx=60)

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
        """
        return -float(collided)


class OFBeeseClass(BeeseClass):
    """
    observation is optic flow data, with a vector (self.get_obs_vector()) appended to every pixel
    the reason we do it this way is because gym.space.Tuple is not supported by stable_baselines3
        we could make our own networks, but just appending is easier
    """

    def __init__(self,
                 client=None,
                 dt=.25,
                 max_tilt=np.pi/18,
                 vehicle_name='',
                 real_time=False,
                 collision_grace=1,
                 initial_position=None,
                 ):
        super().__init__(
            client=client,
            dt=dt,
            max_tilt=max_tilt,
            vehicle_name=vehicle_name,
            real_time=real_time,
            collision_grace=collision_grace,
            initial_position=initial_position,
        )

    def define_observation_space(self):
        """
        defines gym observation space, taking optic flow image and appending a vector to each element
        """
        (C, H, W) = self.get_of_data_shape()
        shape = self.get_obs_shape()
        arr = np.ones(shape)
        low = -np.inf*arr
        low[C:, :, :] = 0
        high = np.inf*arr
        high[C:, :, :] = np.inf

        return gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.float64)

    def get_obs(self):
        of = of_geo(client=self.client, camera_name='front', vehicle_name=self.vehicle_name, FOVx=60)
        C, H, W = of.shape
        # (C,H,W) optic flow data

        vec = self.get_obs_vector()
        # (m,) sized vector, goal conditioning
        obs = np.zeros(self.get_obs_shape())
        # (C+m,H,W)

        obs[:C, :, :] = of
        obs[C:, :, :] = np.expand_dims(vec, axis=(1, 2))  # (m,1,1)

        # places copies of vec at every pixel
        # can technically use gym.spaces.Tuple, and return (of, vec)
        #  unfortunately Tuple space is not supported in stable_baselines
        return obs

    def get_obs_shape(self):
        return (self.get_of_data_shape()[0] + self.get_obs_vector_dim(), *self.get_of_data_shape()[1:])


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
