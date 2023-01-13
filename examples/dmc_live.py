import time
from pathlib import Path

import gym
from cmx import doc
from dm_control import suite
from matplotlib import pyplot as plt

env = gym.make('dmc:Acrobot-swingup-v1')
env.seed(42)

obs = env.reset()

for i in range(100):
    act = env.action_space.sample()
    obs, reward, done, info = env.step(act)
    env.render()
    time.sleep(0.1)

env.close()
doc.flush()
