from art_env.envs import ArtEnv
if __name__ == "__main__":
    env = ArtEnv()
    obs = env.reset()
    env.render()
    for i in range(10):
        action = env.action_space.sample()
        obs, reward,done,_ = env.step(action)
        env.render()