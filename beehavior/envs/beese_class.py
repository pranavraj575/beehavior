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

    def __init__(self, client=None, dt=.25, max_tilt=np.pi/18, real_time=False):
        """
        Args:
            client: airsim interface client
                if None, makes own client
            dt: actions run for this amount of time
            max_tilt: maximum RADIANS that agent can roll/pitch
            real_time: if true, does not pause simulation after each step
        """
        super().__init__()
        self.client = connect_client(client=client)

        self.dt = dt
        self.max_tilt = max_tilt
        self.real_time = real_time

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
                                                                         ),
             pause_after=not self.real_time,
             )
        r = 0

        # observation, reward, termination, truncation, info
        return self.get_obs(), r, False, False, {}

    def get_obs(self):
        return np.zeros(2)

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        super().reset(seed=seed)
        self.client.reset()
        connect_client(client=self.client)
        # obs, info
        return self.get_obs(), {}

    def close(self):
        super().close()
        disconnect_client(client=self.client)


if __name__ == '__main__':
    pass
