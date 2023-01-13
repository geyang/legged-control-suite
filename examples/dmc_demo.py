from pathlib import Path

import gym
from cmx import doc
from dm_control import suite


with doc:
    env = gym.make('dmc:CartPole-balance_sparse-v1')
    env.reset()
    env.seed(42)
    img = env.render('rgb_array')
    doc.savefig(f"{Path(__file__).stem}/cartpole.png")
