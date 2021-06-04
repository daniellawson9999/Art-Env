import sys
import os
import numpy as np
import gym
from art_env.envs import ArtEnv
import argparse
import pickle
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.callbacks import CheckpointCallback



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num-cpu', default=1, type=int, help='number of cpu threads to use for training')
    parser.add_argument('--new-name', default="ppo2_art_env", type=str, help='name of model to train')
    parser.add_argument('--old-name', default=None, type=str, help='name of model to train from')

    args = parser.parse_args()

    new_model_name = args.new_name
    old_model_name = args.old_name
    num_cpu = args.num_cpu
    timesteps = 2e5
    save_freq = 1e4


    def make_env(rank, seed=0):
        """
        Utility function for multiprocessed env.
        
        :param env_id: (str) the environment ID
        :param num_env: (int) the number of environment you wish to have in subprocesses
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        """
        def _init():
            env = ArtEnv()
            env.seed(seed + rank)
            return env
        set_global_seeds(seed)
        return _init

    #env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    env = ArtEnv()

    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=os.path.join('./tensorboard/ppo2', new_model_name))
   
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=os.path.join("./models", new_model_name, "checkpoints/"))
    model.learn(total_timesteps=int(timesteps), callback = checkpoint_callback)
    model.save(os.path.join('./models', new_model_name))
    
if __name__ == "__main__":
    main()
