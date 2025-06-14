from typing import Optional, Tuple

import numpy as np
from gymnasium.core import ObsType
from sympy.physics.units import speed_of_light

from beehavior.envs.beese_class import OFBeeseClass

FLOWER_LOCS_SIM = (
    (
        dict(X=-9220.000000, Y=-22680.000000, Z=50.000000),
        dict(X=-8370.000000, Y=-22830.000000, Z=70.000000),
        dict(X=-7340.000000, Y=-22630.000000, Z=60.000000),
        dict(X=-6660.000000, Y=-22770.000000, Z=70.000000),
        dict(X=-6130.000000, Y=-22790.000000, Z=60.000000),
    ),
    (
        dict(X=-9620.000000, Y=-22100.000000, Z=50.000000),
        dict(X=-8930.000000, Y=-22100.000000, Z=60.000000),
        dict(X=-7680.000000, Y=-22170.000000, Z=50.000000),
        dict(X=-6860.000000, Y=-22170.000000, Z=60.000000),
        dict(X=-6190.000000, Y=-22130.000000, Z=50.000000),
    ),
    (
        dict(X=-9570.000000, Y=-21440.000000, Z=70.000000),
        dict(X=-8940.000000, Y=-21720.000000, Z=50.000000),
        dict(X=-8390.000000, Y=-21410.000000, Z=70.000000),
        dict(X=-7460.000000, Y=-21540.000000, Z=50.000000),
        dict(X=-6820.000000, Y=-21480.000000, Z=50.000000),
        dict(X=-6230.000000, Y=-21690.000000, Z=60.000000),
    ),
    (
        dict(X=-9310.000000, Y=-21070.000000, Z=50.000000),
        dict(X=-8260.000000, Y=-21060.000000, Z=60.000000),
        dict(X=-7550.000000, Y=-21070.000000, Z=50.000000),
        dict(X=-6570.000000, Y=-21060.000000, Z=70.000000),
    ),
    (
        dict(X=-9520.000000, Y=-20690.000000, Z=70.000000),
        dict(X=-8890.000000, Y=-20660.000000, Z=50.000000),
        dict(X=-8270.000000, Y=-20780.000000, Z=60.000000),
        dict(X=-7660.000000, Y=-20690.000000, Z=60.000000),
        dict(X=-7030.000000, Y=-20690.000000, Z=50.000000),
        dict(X=-6210.000000, Y=-20620.000000, Z=70.000000),
    ),
    (
        dict(X=-9470.000000, Y=-19830.000000, Z=60.000000),
        dict(X=-8890.000000, Y=-20270.000000, Z=70.000000),
        dict(X=-8550.000000, Y=-19610.000000, Z=60.000000),
        dict(X=-8240.000000, Y=-20000.000000, Z=50.000000),
        dict(X=-7770.000000, Y=-19800.000000, Z=70.000000),
        dict(X=-7130.000000, Y=-19680.000000, Z=60.000000),
        dict(X=-6790.000000, Y=-20200.000000, Z=60.000000),
        dict(X=-6190.000000, Y=-19900.000000, Z=70.000000),
    ),
    (
        dict(X=-9510.000000, Y=-19110.000000, Z=50.000000),
        dict(X=-8590.000000, Y=-18740.000000, Z=60.000000),
        dict(X=-8180.000000, Y=-19170.000000, Z=50.000000),
        dict(X=-7120.000000, Y=-19070.000000, Z=60.000000),
        dict(X=-6580.000000, Y=-19160.000000, Z=70.000000),
    ),
    (
        dict(X=-9590.000000, Y=-18360.000000, Z=50.000000),
        dict(X=-8920.000000, Y=-18140.000000, Z=50.000000),
        dict(X=-8180.000000, Y=-18350.000000, Z=50.000000),
        dict(X=-7480.000000, Y=-18350.000000, Z=60.000000),
        dict(X=-6860.000000, Y=-18120.000000, Z=70.000000),
    ),
)

FLOWER_LOCS = tuple(
    tuple(np.array([
        flower['X']/100 + 93.85,
        flower['Y']/100 + 221.45,
        -flower['Z']/100,  # z is negated
        flower.get('radius', 1),
    ])
          for flower in tunnel)
    for tunnel in FLOWER_LOCS_SIM
)
TUNNEL_BOUNDS = (
    # squished tunnel
    (-8.35, -3.15),
    # basic default tunnel with two obstacles
    (-2.1, 2.1),
    # tunnel with one center obstacle and two further obstacles
    (2.9, 8.8),
    # narrow tunnel
    (9.75, 11.85),
    # tunnel with two side obstacles then one center obstacle
    (12.55, 16.5),
    # wide tunnel, one center then two side obstacles
    (17.69, 26.8),
    # side to side walls tunnel
    (28, 35.3),
    # empty tunnel
    (36.52, 41.39),
)
TUNNEL_WALLS = np.array([(TUNNEL_BOUNDS[i][1] + TUNNEL_BOUNDS[i + 1][0])/2
                         for i in range(len(TUNNEL_BOUNDS) - 1)])


def which_tunnel(y_pos):
    """
    which tunnel is y position in
    Args:
        y_pos: y position (float)
    Returns:
        integer for tunnel index
    """
    return np.sum(y_pos > TUNNEL_WALLS)


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
                 landing_positions_by_tunnel=None,
                 landing_speed_goal=.5,
                 station_tau=.5,
                 station_c=.5,
                 timeout=30,
                 bounds=(None, None, None),
                 **kwargs,
                 ):
        """
        Args:
            landing_positions_by_tunnel: list of (lists of (x,y,z,radius) for landing positions)
                one list per tunnel
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
        if landing_positions_by_tunnel is None:
            landing_positions_by_tunnel = FLOWER_LOCS
        # list of (m,4) shaped arrays of landing positions
        self.landing_positions_by_tunnel = tuple(np.array(s) for s in landing_positions_by_tunnel)

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
            [x,y,z,radius] of closest landing position,
            distance to landing (xy distance if ignore_z)
        """
        if position is None:
            pose = self.get_pose()
            position = np.array([pose.position.x_val,
                                 pose.position.y_val,
                                 pose.position.z_val,
                                 ])
        tunnel_idx = which_tunnel(y_pos=position[1])
        landing_position = self.landing_positions_by_tunnel[tunnel_idx]
        if ignore_z:
            dists = np.linalg.norm(landing_position[:, :2] - position[:2].reshape((1, 2)), axis=1)
        else:
            dists = np.linalg.norm(landing_position[:, :3] - position.reshape((1, 3)), axis=1)
        idx = np.argmin(dists)
        return landing_position[idx], dists[idx]

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

        term = self.env_time > self.timeout
        trunc = False
        if term:  # do not need to get pose if it already timed out
            return term, trunc
        pose = self.get_pose()
        term = term or self.out_of_bounds(pose=pose)

        if self.GOAL_LAND_ON and (not term) and collided:
            # somtimes ignore collisisons
            # currently ignores if drone is within a cylinder of radius rad, height 2*rad
            pos = np.array([pose.position.x_val,
                            pose.position.y_val,
                            pose.position.z_val,
                            ])
            landing, distxy = self.closest_landing(ignore_z=True)
            xyz = landing[:3]
            rad = landing[3]
            if (xyz[-1] + rad*2 >= pos[-1]) and (pos[-1] >= xyz[-1]) and (distxy <= rad):
                term = term or collided
            # otherwise, ignore collision
        else:
            # if collided, terminate
            term = term or collided

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


class StationBee(GoalBee):
    """
    Only station keeping subgoal
    """

    def __init__(self,
                 initial_position='OVER_FLOWER',
                 bounds=((-7,np.inf), None, None),
                 timeout=15,
                 **kwargs,
                 ):
        """
        Args:
            **kwargs: keyword arguments for GoalBee, OFBeeseClass and BeeseClass
        """
        if initial_position == 'OVER_FLOWER':
            initial_position = []
            for tunnel_idx, tunnel in enumerate(FLOWER_LOCS):
                tunnel_bounds = TUNNEL_BOUNDS[tunnel_idx]
                for flower in tunnel:
                    x, y, z, r = flower
                    rad = 2*r
                    xbnd = (x - rad, x + rad)
                    ybnd = (max(tunnel_bounds[0], y - rad), min(y + rad, tunnel_bounds[1]))
                    zbnd = (z - 3 - rad, z - 3 + rad)

                    initial_position.append((xbnd, ybnd, zbnd))
            initial_position = {bnds: 1/len(initial_position)
                                for bnds in initial_position}
        super().__init__(
            initial_position=initial_position,
            bounds=bounds,
            timeout=timeout,
            **kwargs,
        )
        self.set_station_keep_goal(activate=True)


if __name__ == '__main__':
    import time

    env = StationBee(dt=.1, action_type=GoalBee.ACTION_ACCELERATION_XY, velocity_bounds=1)
    for _ in range(10):
        env.reset()
        for i in range(0, int(10/env.dt), 1):
            action = env.action_space.sample()
            obs, rwd, term, _, _ = env.step(action=action)
            print(rwd)
            if term:
                print("CRASHED")
                break
    env.close()

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
