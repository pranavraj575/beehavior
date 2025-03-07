from typing import Tuple, Optional
import gymnasium as gym
from gymnasium.core import ActType, ObsType
import numpy as np

from airsim_interface.interface import connect_client, disconnect_client, step


class BeeseClass(gym.Env):
    """
    Base class for controlling the quadrotor in unreal
    state is the current unreal engine state
    observation is a 2 vector of zeros right now
    Actions are roll [-1,1], pitch [-1,1], and thrust [0,1]
    reward of d(s,t)-d(s',t), rewarding getting close to target
    terminates if d(s,t)<1
    """

    def __init__(self,
                 client=None,
                 dt=.25,
                 max_tilt=np.pi/18,
                 vehicle_name='',
                 real_time=False,
                 collision_grace=1,
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
        """
        super().__init__()
        if client is None:
            self.client = connect_client(client=client)
        self.vehicle_name = vehicle_name
        self.dt = dt
        self.max_tilt = max_tilt
        self.real_time = real_time
        self.collision_grace = collision_grace

        self.action_space = gym.spaces.Box(low=np.array([-1, -1, 0]), high=1, shape=(3,), dtype=np.float64)

        # REDEFINE THIS
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64)

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
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
            self.update_recent_colision()
        collided = self.has_collided()
        self.col_cnt = max(0, self.col_cnt - 1)
        r = -int(collided)
        # observation, reward, termination, truncation, info
        return self.get_obs(), r, collided, collided, {}

    def get_obs(self):
        return np.zeros(2)

    def has_collided(self):
        """
        Returns: if agent has collided with anything new since last reset
        """
        return self.last_collision != self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name).time_stamp

    def update_recent_colision(self):
        self.last_collision = self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name).time_stamp

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        super().reset(seed=seed)
        self.client.reset()
        self.client = connect_client(client=self.client)
        self.update_recent_colision()
        self.col_cnt = self.collision_grace
        # obs, info
        return self.get_obs(), {}

    def close(self):
        super().close()
        disconnect_client(client=self.client)


if __name__ == '__main__':
    env = BeeseClass(dt=1)
    env.reset()
    env.step(action=np.array([0., 0., 1.]))
    for i in range(5):
        action = env.action_space.sample()
        r, p, t = action
        print('roll:', r, 'pitch:', p, 'thrust:', t)
        env.step(action=action)
    env.close()
