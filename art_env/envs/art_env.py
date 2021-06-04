import gym
from gym import spaces
import numpy as np
from art_env.envs.utils import load_images_abstract
import matplotlib.pyplot as plt
from gym.utils import seeding

class ArtEnv(gym.Env):
    def __init__(self, image_shape=200, n_images=5, use_fixed_noise=True):
    
        self.n_images = n_images
        self.image_shape = image_shape
        self.images, self.arrays = load_images_abstract(n_images, image_shape)
        # normalize
        self.arrays = self.arrays / 255

        self.use_fixed_noise = use_fixed_noise
        self.fixed_noise = None
        if use_fixed_noise:
            self.fixed_noise = np.random.uniform(low=0.0,high=1.0, size=(n_images, image_shape, image_shape, 3))

        ''' 
            each action is a tuple (i,j,r,g,b)
            where i, j is the index of the pixel to draw
            and r,g,b is the red, green, blue channel values to draw
        '''
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0,high=1.0,shape=(image_shape,image_shape, 3), dtype=np.float32)

        # create canvas (currently created part of image) and target image
        self.canvas = None
        self.target = None

        # optional rendering variables
        self.fig = None
        self.ax = None
        self.pause_time = .001

        # training vars
        self.num_steps = 0

    # reward inspried by https://github.com/megvii-research/ICCV2019-LearningToPaint/blob/master/baseline_modelfree/env.py
    def calculate_reward(self):
        distance = self.calculate_distance()
        reward = (self.last_distance - distance) / (self.init_distance+1e-8)
        self.last_distance = distance
        return reward

    def calculate_distance(self):
        return ((self.canvas - self.target) ** 2).mean()

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        self.num_steps += 1

        # index of pixel to draw
        i = int(action[0] * (self.image_shape - 1)) 
        j = int(action[1] * (self.image_shape - 1))
        # perform action
        self.canvas[i,j,:] = action[2:]

        # calculate reward
        #reward = self.calculate_reward()
        reward = -self.calculate_distance()

        done = False
        if self.num_steps >= (self.image_shape ** 2):
            done = True
        
        info = {}
        #  return obs,reward,done,info
        return self.canvas.copy(), reward, done, info

    def reset(self):
        # reset canvas and target image
        #self.canvas = np.zeros((self.image_shape, self.image_shape, 3), dtype=np.float32)

        image_index = np.random.randint(0,self.n_images)
        self.target = self.arrays[image_index]

        if self.use_fixed_noise:
            self.canvas = self.fixed_noise[image_index]
        else:
            self.canvas = np.random.uniform(low=0.0,high=1.0, size=(self.image_shape, self.image_shape, 3))

        self.num_steps = 0
        # initialize distance to target / prev distance
        self.last_distance = self.init_distance = self.calculate_distance()
        return self.canvas.copy()

    def render(self):
        if self.ax is None:
            self.fig, self.ax = plt.subplots()
            plt.ion()
            plt.show()
        temp = self.canvas
        temp = (self.canvas * 255).astype(int)
        self.ax.imshow(temp)
        plt.draw()
        plt.pause(self.pause_time)

    def close(self):
        if self.ax is not None:
            plt.close()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

if __name__ == "__main__":
    env = ArtEnv()
    obs = env.reset()
    for i in range(10):
        action = env.action_space.sample()
        obs, env.step(action)