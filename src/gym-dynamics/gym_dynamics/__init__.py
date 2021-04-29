from gym.envs.registration import register

register(
    id='Free-Ant-v0',
    entry_point='gym_dynamics.envs:FreeAntEnv',
)

register(
    id='Unicycle-v0',
    entry_point='gym_dynamics.envs:UnicycleEnv',
)