import numpy as np
from beehavior.envs.beese_class import OFBeeseClass


class HiBee(OFBeeseClass):
    """
    simple enviornment that gives the agent reward for holding a certian height
    input is optic flow information AND the current pose of the agent
    clearly this is simple to learn, so we use this as a test of the RL pipeline with the Unreal Engine
    """

    def __init__(self,
                 client=None,
                 dt=.25,
                 max_tilt=np.pi/18,
                 vehicle_name='',
                 real_time=False,
                 collision_grace=1,
                 height_range=(2, 3),
                 initial_position=((-1., 0.), (-1., 1.), (-1., -1.5)),
                 timeout=30,
                 img_history_steps=2,
                 see_of_orientation=True,
                 velocity_ctrl=False,
                 fix_z_to=None,
                 of_mapping=lambda x: np.log(np.clip(x, 10e-3, np.inf)),
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
                         see_of_orientation=see_of_orientation,
                         velocity_ctrl=velocity_ctrl,
                         fix_z_to=fix_z_to,
                         of_mapping=of_mapping,
                         )
        self.ht_rng = height_range
        # shoot for average
        self.ht_tgt = sum(height_range)/2

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

    def get_rwd(self, collided, obs):
        """
        -1 for colliding, .5 for correct height, (0,.5) for incorrect height
        """
        if collided:
            return -1.
        ht = obs[-1, 0, 0]  # grab one example of the agent's height, which is copied across the last channel
        if ht >= self.ht_rng[0] and ht <= self.ht_rng[1]:
            return .5
        else:
            rad = (self.ht_rng[1] - self.ht_rng[0])/2
            offset = abs(ht - self.ht_tgt)  # offset>rad, so offset/rad>1
            return .5/(offset/rad)  # .5/(offset/rad)<.5


if __name__ == '__main__':
    import time

    env = HiBee(dt=.2)
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
