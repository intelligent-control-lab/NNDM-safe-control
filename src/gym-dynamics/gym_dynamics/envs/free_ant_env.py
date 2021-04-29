import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class FreeAntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    '''
    Modified from the gym Ant-v2 env. The original env does not provide 
    x y state of the robot. I don't know why.

    
    self.sim.data.qpos are the positions, 
    with the first 7 element the 3D position (x,y,z) and orientation (quaternion x,y,z,w) of the torso, 
    and the remaining 8 positions are the joint angles.

    self.sim.data.qvel are the velocities, 
    with the first 6 elements the 3D velocity (x,y,z) and 3D angular velocity (x,y,z) 
    and the remaining 8 are the joint velocities.

    The cfrc_ext are the external forces (force x,y,z and torque x,y,z) applied to each of the links at the center of mass. 
    For the Ant, this is 14*6: the ground link, the torso link, and 12 links for all legs (3 links for each leg).

    113 states in total.
    x - 0
    y - 1
    0:15 pos
    15:29 vel
    29:113 cfrc_ext

    For the Humanoid, the observation adds some more fields:

    '''
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)
    
    def predict_reward(self, state, action, pred_state):
        xposbefore = state[0]
        xposafter = pred_state[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(action).sum()
        cfrc_ext = pred_state[29:]
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(cfrc_ext, -1, 1)))
        survive_reward = 1.0

        notdone = np.isfinite(pred_state).all() \
            and pred_state[2] >= 0.2 and pred_state[2] <= 1.0

        pred_reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        return pred_reward

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def sample_action(self):
        return self.action_space.sample()

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5