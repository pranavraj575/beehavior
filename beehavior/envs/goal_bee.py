from typing import Optional, Tuple

import numpy as np
from gymnasium.core import ObsType

from beehavior.envs.beese_class import OFBeeseClass


class GoalBee(OFBeeseClass):
    """
    goal conditioned rl env (generic)
    """

    def __init__(self,
                 client=None,
                 dt=.25,
                 action_bounds=None,
                 vehicle_name='',
                 real_time=False,
                 collision_grace=1,
                 of_cameras='front',
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
                 img_history_steps=2,
                 input_img_space=(OFBeeseClass.INPUT_LOG_OF, OFBeeseClass.INPUT_OF_ORIENTATION,),
                 velocity_bounds=2.,
                 action_type=OFBeeseClass.ACTION_ACCELERATION_XY,
                 fix_z_to=None,
                 of_ignore_angular_velocity=True,
                 input_velocity_with_noise=None,
                 central_strip_width=None,
                 global_actions=False,
                 ):
        """
        Args:
            height_range: height goal
            input_velocity_with_noise: whether to include noisy velocity in input
        """
        self.input_vel_w_noise = input_velocity_with_noise
        super().__init__(
            client=client,
            dt=dt,
            action_bounds=action_bounds,
            vehicle_name=vehicle_name,
            real_time=real_time,
            collision_grace=collision_grace,
            of_cameras=of_cameras,
            initial_position=initial_position,
            timeout=timeout,
            img_history_steps=img_history_steps,
            input_img_space=input_img_space,
            velocity_bounds=velocity_bounds,
            action_type=action_type,
            fix_z_to=fix_z_to,
            of_ignore_angular_velocity=of_ignore_angular_velocity,
            central_strip_width=central_strip_width,
            global_actions=global_actions,
        )
        self.bounds = bounds

    def get_observation_vector_part(self):
        if self.get_obs_vector_dim() == 0:
            raise NotImplementedError
        kinematics = self.client.simGetGroundTruthKinematics(vehicle_name=self.vehicle_name)
        vel = np.array([kinematics.linear_velocity.x_val,
                        kinematics.linear_velocity.y_val,
                        kinematics.linear_velocity.z_val])
        return vel + np.random.normal(scale=self.input_vel_w_noise, size=3)

    def get_goal_vector_part(self):
        raise NotImplementedError

    def get_obs_vector(self):
        """
        get obs vector, including roll, pitch, yaw (yaw is encoded as its sine and cosine components,
            to remove the discontinuity at +-pi). this is not an issue for roll,pitch since they will never get this large
        """
        vcs=[]
        if self.get_observation_part_dim()>0:
            vcs.append(self.get_observation_vector_part())
        if self.get_goal_part_dim()>0:
            vcs.append(self.get_goal_vector_part())
        return np.concatenate(vcs,)


    def get_observation_part_dim(self):
        if self.input_vel_w_noise is not None:
            return 3
        else:
            return 0

    def get_goal_part_dim(self):
        return 0

    def get_obs_vector_dim(self):
        """
        shape of obs vector is (4,)
        """
        return self.get_observation_part_dim() + self.get_goal_part_dim()

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
        return term, trunc

    def get_rwd(self, collided, obs):
        """
        -1 for colliding

        should be conditioned on self.get_goal_vector_part
        """
        info_dic = dict()
        if collided:
            return -1., info_dic
        pose = self.get_pose()
        if self.out_of_bounds(pose=pose):
            return -.5, info_dic

        # self.get_goal_vector_part()

        raise NotImplementedError

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

    env = GoalBee(dt=.2,action_type=GoalBee.ACTION_ROLL_PITCH_THRUST)
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
              )
        if term:
            print("CRASHED")
            break
    env.close()
