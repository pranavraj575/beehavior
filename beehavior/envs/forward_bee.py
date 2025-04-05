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
                 initial_position={
                     ((-5., -1.), yrng, (-1., -1.5)): 1/6
                     for yrng in ((-1., 1.),  # -1.8,1.8
                                  (4.20, 7.0),  # 3.14, 8.8
                                  (10.5, 11.),  # 10,11.7
                                  (13.5, 15.5),  # 12.7,16.5
                                  (18.5, 25.8),  # 17.69, 26.8
                                  (29., 34.),  # 28,35.3
                                  )
                 },
                 timeout=30,
                 bounds=((-7., 27), None, None),
                 goal_x=20.,
                 img_history_steps=2,
                 see_of_orientation=True,
                 velocity_ctrl=True,
                 fix_z_to=None,
                 of_mapping=lambda x: np.log(np.clip(x, 10e-3, np.inf)),
                 of_ignore_angular_velocity=True,
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
                         img_history_steps=img_history_steps,
                         velocity_ctrl=velocity_ctrl,
                         see_of_orientation=see_of_orientation,
                         fix_z_to=fix_z_to,
                         of_mapping=of_mapping,
                         of_ignore_angular_velocity=of_ignore_angular_velocity,
                         )
        self.bounds = bounds
        self.goal_x = goal_x
        self.farthest_reached = None

    def get_obs_vector(self):
        """
        get obs vector, including roll, pitch, yaw (yaw is encoded as its sine and cosine components,
            to remove the discontinuity at +-pi). this is not an issue for roll,pitch since they will never get this large
        """
        # TODO: ignore rpy
        pose = self.get_pose()
        ht = -pose.position.z_val
        r, p, y = self.get_orientation_eulerian(quaternion=(pose.orientation.x_val,
                                                            pose.orientation.y_val,
                                                            pose.orientation.z_val,
                                                            pose.orientation.w_val,
                                                            ))
        cy = np.cos(y)
        sy = np.sin(y)
        return np.array([
            r,
            p,
            cy,
            sy,
            # ht,
        ])

    def get_obs_vector_dim(self):
        """
        shape of obs vector is (4,)
        """
        return 4

    def out_of_bounds(self, pose):

        for val, t in zip(
                (pose.position.x_val, pose.position.y_val, pose.position.z_val),
                self.bounds,
        ):
            if t is None:
                continue
            low, high = t
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
        pose = self.get_pose()
        term = self.out_of_bounds(pose=pose)
        if pose.position.x_val > self.goal_x:
            term = True
        return term, trunc

    def get_rwd(self, collided, obs):
        """
        -1 for colliding, .5 for correct height, (0,.5) for incorrect height
        """
        info_dic = {'succ': False}
        if collided:
            return -1., info_dic
        pose = self.get_pose()
        if self.out_of_bounds(pose=pose):
            return -.5, info_dic
        if pose.position.x_val > self.goal_x:
            info_dic['succ'] = True
            return 1., info_dic

        if pose.position.x_val > self.farthest_reached:
            val = pose.position.x_val - self.farthest_reached
        else:
            val = 0
        self.farthest_reached = max(
            self.farthest_reached,
            pose.position.x_val,
        )
        return val, info_dic

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        stuff = super().reset(seed=seed, options=options, )
        self.farthest_reached = self.get_pose().position.x_val
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
