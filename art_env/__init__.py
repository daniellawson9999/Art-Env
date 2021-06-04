from gym.envs.registration import register

register(
    id='art-env-v0',
    entry_point='art_env.envs:ArtEnv',
    kwargs={
        'image_shape':200, 'n_images':50, 'use_fixed_noise':True
    }
    
)