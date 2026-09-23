from gymnasium.envs.registration import register

register(id='NeuroRacerDiscrete-v0', entry_point='neuroracer_gym.env:NeuroRacerEnv', max_episode_steps=1000)
register(id='NeuroRacerContinuous-v0', entry_point='neuroracer_gym.env:NeuroRacerEnv', max_episode_steps=1000,
         kwargs={'continuous': True})
