import os
import numpy as np
import gym
import pandas as pd
import progressbar
import gym_dynamics
class GymDynamicsDataset(object):
    def __init__(self, env_name, size=100000):
        self.env_name = env_name
        self.size = size
        self.data_path = '../data/'+self.env_name+'_'+str(self.size)+'.zip'
        self.load_dataset()

    def load_dataset(self):
        if not os.path.exists(self.data_path):
            self.generate_dataset()
        self.df = pd.read_pickle(self.data_path)
        
    def generate_dataset(self):
        env = gym.make(self.env_name)

        state = env.reset()
        step = 0

        df = pd.DataFrame(columns=['state_action', 'dot_state', 'reward', 'done'])
        print("Generating dynamics data")
        for step in progressbar.progressbar(range(self.size)):
        # for step in range(self.size):
            action = env.sample_action()
            new_state, reward, done, info = env.step(action)
            df.loc[step] = [np.hstack([state, action]).astype(np.float32), info["dot_state"].astype(np.float32), reward, done]
            state = new_state
            if done:
                state = env.reset()
        
        df.to_pickle(self.data_path, compression='zip')

    def __getitem__(self, idx):
        return self.df.iat[idx, 0], self.df.iat[idx, 1]

    def __len__(self):
        return len(self.df)


class UnicycleDynamicsDataset(object):
    def __init__(self, env_name, size=1000):
        self.env_name = env_name
        self.size = size
        self.data_path = '../data/Unicycle-v0_'+str(size)+'.zip'
        self.load_dataset()

    def load_dataset(self):
        if not os.path.exists(self.data_path):
            self.generate_dataset()
        self.df = pd.read_pickle(self.data_path)
        
    def generate_dataset(self):
        env = UnicycleEnv()

        state = env.reset()
        step = 0

        df = pd.DataFrame(columns=['state_action', 'state_residual'])
        print("Generating dynamics data")
        for step in progressbar.progressbar(range(self.size)):
            action = env.sample_action()
            new_state = env.step(action)
            df.loc[step] = [np.vstack([state, action]).astype(np.float32), (new_state-state).astype(np.float32)]
            state = new_state
            if step % 1000 == 0:
                state = env.reset()
            env.render()
        
        df.to_pickle(self.data_path, compression='zip')

    def __getitem__(self, idx):
        return self.df.iat[idx, 0], self.df.iat[idx, 1]

    def __len__(self):
        return len(self.df)


if __name__ == '__main__':
    
    # DynamicsDataset("Ant-v2")
    # UnicycleDynamicsDataset()
    # GymDynamicsDataset("Free-Ant-v0")
    GymDynamicsDataset("Unicycle-v0")