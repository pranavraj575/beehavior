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
        flower.get('radius', .6),
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
    GOAL_FORWARD = 'fwd'
    GOAL_HOVER = 'hvr'
    GOAL_STATION_KEEP = 'stn'
    GOAL_LAND_ON = 'lnd'

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
                 initial_goals=(GOAL_FORWARD,),
                 goal_x=20.,
                 station_tau=.5,
                 station_c=.5,
                 timeout=30,
                 bounds=((-7, np.inf), None, None),
                 **kwargs,
                 ):
        """
        Args:
            initial_position: if 'OVER_FLOWER', spawns over random flower TODO: sometimes spawns within obstacle
            landing_positions_by_tunnel: list of (lists of (x,y,z,radius) for landing positions)
                one list per tunnel
                collisions ignored if within radius of landing position
            initial_goals: goals to initialize enviornment with, either iterable, or dict (goal -> weight)
            goal_x: if GOAL_FORWARD, stops episode when this x value is reached
            station_tau: for GOAL_STATION_KEEPING, tau from https://www.nature.com/articles/s41586-020-2939-8
            station_c: for GOAL_STATION_KEEPING, c_cliff from https://www.nature.com/articles/s41586-020-2939-8
            dt: also used to calculate reward for GOAL_HOVER
            velocity_bounds: also used to calculate reward for GOAL_HOVER
            **kwargs: keyword arguments for OFBeeseClass and BeeseClass
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
                    zbnd = (z - 1, z - 1 - 2*rad)
                    initial_position.append((xbnd, ybnd, zbnd))
            initial_position = {bnds: 1/len(initial_position)
                                for bnds in initial_position}
        super().__init__(
            initial_position=initial_position,
            timeout=timeout,
            **kwargs,
        )
        self.bounds = bounds
        self.goal_x = goal_x
        if landing_positions_by_tunnel is None:
            landing_positions_by_tunnel = FLOWER_LOCS
        # list of (m,4) shaped arrays of landing positions
        self.landing_positions_by_tunnel = tuple(np.array(s) for s in landing_positions_by_tunnel)

        # self.landing_speed_goal = landing_speed_goal

        self.station_c = station_c
        self.station_tau = station_tau
        if type(initial_goals) == dict:
            self.initial_goals = initial_goals.copy()
        else:
            self.initial_goals = {g: 1 for g in initial_goals}
        self.active_goals = self.initial_goals.copy()
        self.rwds = dict()

    def set_forward_goal(self, weight=1):
        """
        set forward goal
        Args:
            weight: weight of forward goal, 0 if deactivate
        """
        if weight:
            # reset the furthest agent has traveled to current position
            self.farthest_reached = self.get_pose().position.x_val
        self.activate_goal(goal=self.GOAL_FORWARD, weight=weight)

    def set_hover_goal(self, weight=1):
        """
        set hover goal
        Args:
            weight: weight of hover goal, 0 if deactivate
        """
        if weight:
            self.old_pose = self.get_pose()
        self.activate_goal(goal=self.GOAL_HOVER, weight=weight)

    def set_station_keep_goal(self, weight=1):
        """
        set station keeping goal
        Args:
            weight: weight of station keeping goal, 0 if deactivate
        """
        self.activate_goal(goal=self.GOAL_STATION_KEEP, weight=weight)

    def set_land_goal(self, weight=1):
        """
        set landing goal
        Args:
            weight: weight of landing goal, 0 if deactivate
        """
        self.activate_goal(goal=self.GOAL_LAND_ON, weight=weight)

    def activate_goal(self, goal, weight):
        """
        set weight of a goal
        Args:
            weight: weight to use
            goal: goal to set
        Returns:

        """
        if weight:
            self.active_goals[goal] = weight
        else:
            self.active_goals.pop(goal)

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

    def get_observation_vector_part(self):
        raise NotImplementedError

    def get_goal_part_dim(self):
        return 1 + 1 + 1 + 1

    def get_goal_vector_part(self):
        return np.array([self.active_goals.get(g, 0.)
                         for g in (self.GOAL_FORWARD,
                                   self.GOAL_HOVER,
                                   self.GOAL_STATION_KEEP,
                                   self.GOAL_LAND_ON,
                                   )
                         ],
                        dtype=np.float64,
                        )
        return np.array([g in self.active_goals
                         for g in (self.GOAL_FORWARD,
                                   self.GOAL_HOVER,
                                   self.GOAL_STATION_KEEP,
                                   self.GOAL_LAND_ON,
                                   )
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

    def within_landing_area(self, position=None):
        """
        checks if drone is within a specified landing area
            currently if drone is within a cylinder of radius rad, height 2*rad above the target
        Args:
            position:
        Returns:
        """
        if position is None:
            pose = self.get_pose()
            position = np.array([pose.position.x_val,
                                 pose.position.y_val,
                                 pose.position.z_val,
                                 ])
        landing, distxy = self.closest_landing(position=position,
                                               ignore_z=True,
                                               )
        land_xyz = landing[:3]
        rad = landing[3]

        # z values are inverted for some reason
        z = -position[-1]
        z_land = -land_xyz[2]
        return (z_land + rad*2 >= z) and (z >= z_land) and (distxy <= rad)

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
        if self.GOAL_FORWARD in self.active_goals:
            term = term or (pose.position.x_val >= self.goal_x)

        if (self.GOAL_LAND_ON in self.active_goals) and (not term) and collided:
            # somtimes ignore collisisons, if drone is about to land
            position = np.array([pose.position.x_val,
                                 pose.position.y_val,
                                 pose.position.z_val,
                                 ])
            if self.within_landing_area(position=position):
                if self.drone_landed(collided=collided, position=position):
                    term = True
            else:
                # if not within landing area, this is a collision
                term = term or collided
        else:
            # otherwise, if collided, terminate
            term = term or collided

        return term, trunc

    def drone_landed(self, collided, position=None):
        return self.within_landing_area(position=position) and collided

    def get_rwd(self, collided, obs):
        """
        -1 for colliding

        should be conditioned on self.get_goal_vector_part
        """
        info_dic = dict()

        if (self.GOAL_LAND_ON not in self.active_goals) and collided:
            # for every other goal, collsion is bad
            return -1., info_dic
        pose = self.get_pose()
        if self.out_of_bounds(pose=pose):
            return -.5, info_dic

        # self.get_goal_vector_part()
        if self.GOAL_FORWARD in self.active_goals:
            new_furthest = max(self.get_pose().position.x_val, self.farthest_reached)
            fwd_rwd = new_furthest - self.farthest_reached
            self.farthest_reached = new_furthest
            self.rwds[self.GOAL_FORWARD] = fwd_rwd
            if self.farthest_reached >= self.goal_x:
                info_dic['succ'] = True

        if self.GOAL_HOVER in self.active_goals:
            dp = np.array([
                pose.position.x_val - self.old_pose.position.x_val,
                pose.position.y_val - self.old_pose.position.y_val,
                pose.position.z_val - self.old_pose.position.z_val,
            ])
            hover_rwd = (self.velocity_bounds*self.dt - np.linalg.norm(dp))

            self.rwds[self.GOAL_HOVER] = hover_rwd
            self.old_pose = pose

        if self.GOAL_STATION_KEEP in self.active_goals:
            landing, dist = self.closest_landing(position=np.array([pose.position.x_val,
                                                                    pose.position.y_val,
                                                                    pose.position.z_val,
                                                                    ]),
                                                 ignore_z=True,
                                                 )

            if dist < landing[-1]:
                station_rwd_shape = 1
            else:
                station_rwd_shape = self.station_c*np.exp2(-(dist - landing[-1])/self.station_tau)

            station_rwd = station_rwd_shape
            self.rwds[self.GOAL_STATION_KEEP] = station_rwd

        if self.GOAL_LAND_ON in self.active_goals:
            position = np.array([
                pose.position.x_val,
                pose.position.y_val,
                pose.position.z_val,
            ])
            if collided and (not self.within_landing_area(position=position)):
                # collided outside landing area
                return -1., info_dic
            # TODO THIS
            # https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py

            landing_rwd_shape = 0.

            # amount to weight station keeping part of reward
            lnd_station_keeping_c = .2
            # amount to weight keeping speed low
            lnd_speed_c = 1
            # amount to weight distance
            lnd_dist_c = 1
            # amount to weight angle flatness
            lnd_angle_c = 1
            # amount to weight leg touching down
            lnd_contact_c = .1

            # success bonus
            lnd_success_c = 3.

            # amnt to penalize staying in air
            air_penalty = -.05

            position = np.array([pose.position.x_val,
                                 pose.position.y_val,
                                 pose.position.z_val,
                                 ])
            kinematics = self.get_kinematics()
            velocity = np.array([kinematics.linear_velocity.x_val,
                                 kinematics.linear_velocity.y_val,
                                 kinematics.linear_velocity.z_val,
                                 ])

            roll, pitch, yaw = self.get_orientation_eulerian(pose=pose)
            robot_angle = abs(roll) + abs(pitch)  # dont care about yaw
            leg_ground_contact = collided  # this is true if drone has collided with something and drone is within landing region

            landing, dist_xy = self.closest_landing(position=position,
                                                    ignore_z=True,
                                                    )
            rad = landing[-1]
            dist_xyz = np.linalg.norm(landing[:3] - position)

            f = np.array([
                dist_xyz/rad,  # penalize distance from landing
                np.linalg.norm(velocity),  # penalize moving quickly
                robot_angle,  # penalize angle
                leg_ground_contact,  # reward leg touching ground
            ])
            w = np.array([-lnd_dist_c,
                          -lnd_speed_c,
                          -lnd_angle_c,
                          lnd_contact_c,
                          ])
            landing_rwd_shape += np.dot(f, w)

            land_bonus = 0

            if self.drone_landed(collided=collided, position=position):
                land_bonus = lnd_success_c*np.exp(-(dist_xy/rad)**2)
            self.rwds[self.GOAL_LAND_ON] = (air_penalty +
                                            (landing_rwd_shape - self.past_goal_shape.get(self.GOAL_LAND_ON,
                                                                                          landing_rwd_shape)) +
                                            land_bonus)
            self.past_goal_shape[self.GOAL_LAND_ON] = landing_rwd_shape
        # weighted sum
        rwd = sum(self.rwds[goal]*self.active_goals.get(goal, 0)
                  for goal in self.rwds)
        return rwd, info_dic

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
        self.rwds = {k: 0.
                     for k in (self.GOAL_FORWARD,
                               self.GOAL_HOVER,
                               self.GOAL_STATION_KEEP,
                               self.GOAL_LAND_ON,
                               )
                     }
        self.past_goal_shape = dict()
        self.active_goals = self.initial_goals.copy()
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
            initial_goals=(GoalBee.GOAL_FORWARD,),
            **kwargs,
        )
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
                 timeout=15,
                 **kwargs,
                 ):
        """
        Args:
            **kwargs: keyword arguments for GoalBee, OFBeeseClass and BeeseClass
        """
        super().__init__(
            initial_position=initial_position,
            timeout=timeout,
            initial_goals=(GoalBee.GOAL_STATION_KEEP,),
            **kwargs,
        )


class LandingBee(GoalBee):
    """
    Only landing subgoal
    """

    def __init__(self,
                 initial_position='OVER_FLOWER',
                 timeout=15,
                 **kwargs,
                 ):
        """
        Args:
            **kwargs: keyword arguments for GoalBee, OFBeeseClass and BeeseClass
        """
        super().__init__(
            initial_position=initial_position,
            timeout=timeout,
            initial_goals=(GoalBee.GOAL_LAND_ON,),
            **kwargs,
        )


if __name__ == '__main__':
    import time

    env = LandingBee(dt=.1, action_type=GoalBee.ACTION_VELOCITY)
    env.reset(options={'initial_pos': env.landing_positions_by_tunnel[1][0][:3] + (-2, 0, -.6)})
    for i in range(0, int(15/env.dt), 1):
        action = np.array([.25, 0, .025])
        obs, rwd, term, _, _ = env.step(action=action)
        print(rwd)
        print('within landing', env.within_landing_area())
        print()
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
