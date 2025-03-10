import numpy as np
from beehavior.envs.beese_class import BeeseClass


class HiBee(BeeseClass):
    def __init__(self,
                 client=None,
                 dt=.25,
                 max_tilt=np.pi/18,
                 vehicle_name='',
                 real_time=False,
                 collision_grace=1,
                 height_range=(2, 3),
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
                         )
        self.ht_rng = height_range

    def get_obs_vector(self):
        """
        get height, a single real number
        """
        pose = self.get_pose()
        return np.array([-pose.position.z_val])

    def get_obs_vector_dim(self):
        """
        shape of obs vector is (1,)
        """
        return 1

    def get_rwd(self, collided, obs):
        """
        -1 for colliding, .5 for correct height, 0 for incorrect height
        """
        if collided:
            return -1.
        ht = obs[0, 0, -1]  # grab one example of the agent's height
        if ht > self.ht_rng[0] and ht < self.ht_rng[1]:
            return .5
        else:
            return 0.

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
              'height:',obs[0,0,-1],
              )
        if term:
            print("CRASHED")
            break
    env.close()