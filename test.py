import gym
from art_env.envs import ArtEnv
import matplotlib.pyplot as plt
import numpy as np
import sys
from stable_baselines import PPO2


def main(model_path):
    num_steps = 1000
    
    env = ArtEnv()
 
    # in function imports to fix env problem, normally move to top of script
    from stable_baselines.common.policies import MlpPolicy


    model = PPO2.load(model_path)
    
    episode_rewards = [0.0]
    obs = env.reset()
    done = False
    env.render()
    for i in range(num_steps):
        # model.prection returns a tuple w/ (action, None) for some reason
        action = model.predict(obs)[0]
        obs, reward, done, _ = env.step(action)
        env.render()
        episode_rewards.append(reward)
        if done:
            obs = env.reset()

    env.close()       
    fig, ax = plt.subplots()
    ax.plot(episode_rewards)
    plt.show()
   


if __name__ == "__main__":
    assert (len(sys.argv) == 2), "python test.py [model_path]"
    model_path = sys.argv[1]
    main(model_path)
    
 

