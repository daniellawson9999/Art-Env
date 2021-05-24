import gym
from gym import Spaces
import numpy as np


class ArtEnv(gym.Env):
    def __init__(self,):
    
        ''' 
            each action is a tuple (i,j,r,g,b)
            where i, j is the index of the pixel to draw
            and r,g,b is the red, green, blue channel values to draw
        '''
        self.action_space = Spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
        self.observation_space = 

    def step(self, action):
        #  return obs,reward,done,info
        pass 

    def reset(self):
        # return obs
        pass

    def render(self):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    pass