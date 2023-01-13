from pathlib import Path

import gym
from cmx import doc
from dm_control import suite
from matplotlib import pyplot as plt

with doc:
    env = gym.make('dmc:Acrobot-swingup-v1')
    env.reset()
    env.seed(42)
    img = env.render('rgb_array')
    doc.figure(img, f"{Path(__file__).stem}/cartpole.png", zoom="400%", title="Cartpole", caption="""
This is a dm_control environment called Cartpole.
    """)

env.close()
doc.flush()
