from typing import Optional, Tuple

import numpy as np
from gymnasium.core import ObsType

from beehavior.envs.beese_class import OFBeeseClass


class ForwardBee(OFBeeseClass):
    """
    reward for moving forward (x direction) through obstacles
    """

    def __init__(self,
                 client=None,
                 dt=.25,
                 max_tilt=np.pi/18,
                 vehicle_name='',
                 real_time=False,
                 collision_grace=1,
                 initial_position=((-1., 0.), (-1., 1.), (-1., -1.5)),
                 timeout=30,
                 bounds=((-5., 27), (-2.5, 2.5), (-5., 0.)),
                 goal_x=24.,
                 img_stack_size=3,
                 velocity_ctrl=True,
                 fix_z_to=None,
                 ):
        """
        Args:
            height_range: height goal
        """
        super().__init__(client=client,
                         dt=dt,
                         max_tilt=max_tilt,
                         vehicle_name=vehicle_name,
                         real_time=real_time,
                         collision_grace=collision_grace,
                         initial_position=initial_position,
                         timeout=timeout,
                         img_stack_size=img_stack_size,
                         velocity_ctrl=velocity_ctrl,
                         fix_z_to=fix_z_to,
                         )
        self.bounds = bounds
        self.goal_x = goal_x
        self.farthest_reached = None

    def get_obs_vector(self):
        """
        get height, a single real number
        """
        pose = self.get_pose()
        ht = -pose.position.z_val
        r, p, y = self.get_orientation_eulerian(quaternion=(pose.orientation.x_val,
                                                            pose.orientation.y_val,
                                                            pose.orientation.z_val,
                                                            pose.orientation.w_val,
                                                            ))

        return np.array([
            r,
            p,
            y,
            ht,
        ])

    def get_obs_vector_dim(self):
        """
        shape of obs vector is (4,)
        """
        return 4

    def out_of_bounds(self, pose):

        for val, (low, high) in zip(
                (pose.position.x_val, pose.position.y_val, pose.position.z_val),
                self.bounds,
        ):
            if val < low or val > high:
                return True
        return False

    def get_termination(self, collided):
        """
        terminate if out of bounds or in goal region
        """
        term, trunc = super().get_termination(collided=collided)
        if term:
            return term, trunc
        pose = self.client.simGetVehiclePose(vehicle_name=self.vehicle_name)
        term = self.out_of_bounds(pose=pose)
        if pose.position.x_val > self.goal_x:
            term = True
        return term, trunc

    def get_rwd(self, collided, obs):
        """
        -1 for colliding, .5 for correct height, (0,.5) for incorrect height
        """
        if collided:
            return -1.
        pose = self.client.simGetVehiclePose(vehicle_name=self.vehicle_name)

        if self.out_of_bounds(pose=pose):
            return -.5
        if pose.position.x_val > self.goal_x:
            return 10.

        if pose.position.x_val > self.farthest_reached:
            val = pose.position.x_val - self.farthest_reached
        else:
            val = 0
        self.farthest_reached = max(
            self.farthest_reached,
            pose.position.x_val,
        )
        return val

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        stuff = super().reset(seed=seed, options=options, )
        self.farthest_reached = self.client.simGetVehiclePose(vehicle_name=self.vehicle_name).position.x_val
        return stuff


if __name__ == '__main__':
    import time

    env = ForwardBee(dt=.2)
    env.reset()
    env.step(action=np.array([0., 0., 1.]))
    for _ in range(0, int(.5/env.dt), 1):
        env.step(action=np.array([0., 0., 1.]))
    for i in range(0, int(10/env.dt), 1):
        action = env.action_space.sample()
        r, p, t = action
        obs, rwd, term, _, _ = env.step(action=action)
        print('roll:', r,
              'pitch:', p,
              'thrust:', t,
              'reward:', rwd,
              'height:', obs[0, 0, -1],
              )
        if term:
            print("CRASHED")
            break
    env.close()
