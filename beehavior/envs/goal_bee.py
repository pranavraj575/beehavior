from typing import Optional, Tuple

import numpy as np
from gymnasium.core import ObsType
from sympy.physics.units import speed_of_light

from beehavior.envs.beese_class import OFBeeseClass


class GoalBee(OFBeeseClass):
    """
    goal conditioned rl env
    IMPLEMENTED GOALS:
        move forward:
            reward is the amount agent advances the furthest distance travelled in +x direction
                if agent does not increase the furthest it has traveled, reward is 0
        hover:
            reward is (c-|movement|) at each step.
                c = self.velocity_bounds*self.dt, theoretic max distance traveled if agent moves at velocity_bounds
        station_keeping:
            reward modeled after RL balloon station keeping task
        landing:
            reward modeled after gymnasium Lunar Lander task
    """

    def __init__(self,
                 initial_position={
                     ((-5., -1.), yrng, (-2., -6.9)): 1/6
                     for yrng in ((-1., 1.),  # -1.8,1.8
                                  (4.20, 7.0),  # 3.14, 8.8
                                  (10.5, 11.),  # 10,11.7
                                  (13.5, 15.5),  # 12.7,16.5
                                  (18.5, 25.8),  # 17.69, 26.8
                                  (29., 34.),  # 28,35.3
                                  )
                 },
                 landing_positions=((0, 0, 1, 3),),
                 landing_speed_goal=.5,
                 station_tau=1,
                 station_c=.5,
                 timeout=30,
                 bounds=((-7., 27), None, None),
                 **kwargs,
                 ):
        """
        Args:
            landing_positions: list of (x,y,z,radius) for landing positions
                collisions ignored if within radius of landing position
            landing_speed_goal: want to land at or below this speed
            station_tau: for GOAL_STATION_KEEPING, tau from https://www.nature.com/articles/s41586-020-2939-8
            station_c: for GOAL_STATION_KEEPING, c_cliff from https://www.nature.com/articles/s41586-020-2939-8
            dt: also used to calculate reward for GOAL_HOVER
            velocity_bounds: also used to calculate reward for GOAL_HOVER
            **kwargs: keyword arguments for OFBeeseClass and BeeseClass
        """
        super().__init__(
            initial_position=initial_position,
            timeout=timeout,
            **kwargs,
        )
        self.bounds = bounds

        self.landing_positions = np.array(landing_positions)  # (m,4) shaped array of landing positions
        self.landing_speed_goal = landing_speed_goal

        self.station_c = station_c
        self.station_tau = station_tau

        self.GOAL_FORWARD = False
        self.GOAL_HOVER = False
        self.GOAL_STATION_KEEP = False
        self.GOAL_LAND_ON = False

    def set_forward_goal(self, activate=True):
        """
        Args:
            activate: whether to activate the forward goal
        """
        if activate:
            # reset the furthest agent has traveled to current position
            self.farthest_reached = self.get_pose().position.x_val
        self.GOAL_FORWARD = activate

    def set_hover_goal(self, activate=True):
        """
        Args:
            activate: whether to activate the hover goal
        """
        if activate:
            self.old_pose = self.get_pose()
        self.GOAL_HOVER = activate

    def set_station_keep_goal(self, activate=True):
        """
        Args:
            activate: whether to activate the station keeping goal
        """
        self.GOAL_STATION_KEEP = activate

    def closest_landing(self, position=None, ignore_z=True):
        """
        gets closest landing position to specified (x,y,z) position
        Args:
            position: (x,y,z) np array of position to check, if None, uses self.pose
            ignore_z: ignore z direction when calculating closest
        Returns:
            [x,y,z,radius] of closest landing position, distance to landing (xy distance if ignore_z)
        """
        if position is None:
            pose = self.get_pose()
            position = np.array([pose.position.x_val,
                                 pose.position.y_val,
                                 pose.position.z_val,
                                 ])
        if ignore_z:
            dists = np.linalg.norm(self.landing_positions[:, :2] - position.view((1, 2)), axis=1)
        else:
            dists = np.linalg.norm(self.landing_positions[:, :3] - position.view((1, 3)), axis=1)
        idx = np.argmin(dists)
        return self.landing_positions[idx], dists[idx]

    def set_land_goal(self, activate=True):
        """
        Args:
            activate: whether to activate the land goal
        """
        self.GOAL_LAND_ON = activate

    def get_observation_vector_part(self):
        raise NotImplementedError

    def get_goal_part_dim(self):
        return 1 + 1 + 1 + 1

    def get_goal_vector_part(self):
        return np.array([self.GOAL_FORWARD,
                         self.GOAL_HOVER,
                         self.GOAL_STATION_KEEP,
                         self.GOAL_LAND_ON,
                         ],
                        dtype=np.float64,
                        )

    def get_obs_vector(self):
        """
        get obs vector, including roll, pitch, yaw (yaw is encoded as its sine and cosine components,
            to remove the discontinuity at +-pi). this is not an issue for roll,pitch since they will never get this large
        """
        vcs = []
        if self.get_observation_part_dim() > 0:
            vcs.append(self.get_observation_vector_part())
        if self.get_goal_part_dim() > 0:
            vcs.append(self.get_goal_vector_part())
        return np.concatenate(vcs, )

    def get_observation_part_dim(self):
        return 0

    def get_obs_vector_dim(self):
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
        rwd = 0
        if self.GOAL_FORWARD:
            new_furthest = max(self.get_pose().position.x_val, self.farthest_reached)
            rwd += new_furthest - self.farthest_reached
            self.farthest_reached = new_furthest

        if self.GOAL_HOVER:
            pose = self.get_pose()
            dp = np.array([
                pose.position.x_val - self.old_pose.position.x_val,
                pose.position.y_val - self.old_pose.position.y_val,
                pose.position.z_val - self.old_pose.position.z_val,
            ])
            rwd += (self.velocity_bounds*self.dt - np.linalg.norm(dp))

            self.old_pose = pose
        if self.GOAL_STATION_KEEP:
            landing, dist = self.closest_landing(position=np.array([pose.position.x_val,
                                                                    pose.position.y_val,
                                                                    pose.position.z_val,
                                                                    ]),
                                                 ignore_z=True,
                                                 )

            if dist < landing[-1]:
                rwd += 1
            else:
                rwd += self.station_c*np.exp2(-(dist - landing[-1])/self.station_tau)

        if self.GOAL_LAND_ON:
            # TODO THIS
            # https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py

            # reward shaping
            landing_rwd_shape = 0.

            # amount to weight station keeping part of reward
            lnd_station_keeping_c = .2
            # amount to weight keeping speed low
            lnd_speed_c = .1
            # amount to weight distance
            lnd_dist_c = .1
            # amount to weight angle flatness
            lnd_angle_c = .1
            # amount to weight leg touching down
            lnd_contact_c = .1

            # amnt to penalize staying in air
            air_penalty = -.1

            position = np.array([pose.position.x_val,
                                 pose.position.y_val,
                                 pose.position.z_val,
                                 ])
            kinematics = self.get_kinematics()
            velocity = np.array([kinematics.linear_velocity.x_val,
                                 kinematics.linear_velocity.y_val,
                                 kinematics.linear_velocity.z_val,
                                 ])

            robot_angle = 0
            leg_ground_contact = 0

            landing, distxy = self.closest_landing(position=position,
                                                   ignore_z=True,
                                                   )
            dist_xyz = np.linalg.norm(landing[:3] - position)

            # penalize distance from landing
            landing_rwd_shape += -lnd_dist_c*dist_xyz

            # penalize moving quickly
            landing_rwd_shape += -lnd_speed_c*np.linalg.norm(velocity)

            # penalize angle
            landing_rwd_shape += -lnd_angle_c*robot_angle

            # reward leg touching ground
            landing_rwd_shape += -lnd_contact_c*leg_ground_contact

            rwd += landing_rwd_shape - self.past_landing_rwd_shape
            rwd += air_penalty
            self.past_landing_rwd_shape = landing_rwd_shape
        return rwd

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        stuff = super().reset(seed=seed,
                              options=options,
                              )
        self.old_pose = self.get_pose()
        self.farthest_reached = self.get_pose().position.x_val
        self.past_landing_rwd_shape = 0.
        return stuff


class FwdGoalBee(GoalBee):
    """
    Forward bee as subclass of goal bee
    """

    def __init__(self,
                 goal_x=20.,
                 **kwargs,
                 ):
        """
        Args:
            goal_x:
            **kwargs: keyword arguments for GoalBee, OFBeeseClass and BeeseClass
        """
        super().__init__(
            **kwargs,
        )
        self.set_forward_goal(activate=True)
        self.goal_x = goal_x

    def get_termination(self, collided):
        """
        terminate if out of bounds or in goal region
        """
        term, trunc = super().get_termination(collided=collided)
        if term:
            return term, trunc
        pose = self.get_pose()
        term = pose.position.x_val >= self.goal_x
        return term, trunc


if __name__ == '__main__':
    import time

    env = GoalBee(dt=.1, action_type=GoalBee.ACTION_ROLL_PITCH_THRUST)
    env.reset()
    env.step(action=np.array([0., 0., 1.]))
    for i in range(0, int(10/env.dt), 1):
        action = env.action_space.sample()
        obs, rwd, term, _, _ = env.step(action=action)

        if term:
            print("CRASHED")
            break
    env.close()
    print({
        k: obs[k].shape for k in obs
    })
