from gym.envs.registration import register

register(
    id='art-env-v0',
    entry_point='art_env.envs:ArtEnv'
)