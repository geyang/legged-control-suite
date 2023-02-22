import os
from pathlib import Path
import tempfile

import numpy as np
import imageio
from skimage import img_as_ubyte
import dm_control
from dm_control.mujoco.wrapper import core
from dm_control.mujoco.wrapper.mjbindings import mjlib
import gym

import matplotlib.pyplot as plt

from cmx import doc


"""
Original masses (volume * density)
0  world [ 0       ]
1  torso [ 9.46    ]
2 bthigh [ 2.34    ]
3  bshin [ 2.4     ]
4  bfoot [ 1.66    ]
5 fthigh [ 2.18    ]
6  fshin [ 1.82    ]
7  ffoot [ 1.34    ]
Total: 21.183717375253927


Masses scaled to total mass of 14kg
0  world [ 0       ]
1  torso [ 6.25    ]
2 bthigh [ 1.54    ]
3  bshin [ 1.59    ]
4  bfoot [ 1.1     ]
5 fthigh [ 1.44    ]
6  fshin [ 1.2     ]
7  ffoot [ 0.885   ]
"""

doc @ "# Paramcheetah environment hacking"


with doc:
    env = gym.make('lcs:Paramcheetah-run-v1')
    env.seed(42)
    env.reset()

    doc.print(f'Observation space dim: {env.observation_space.shape}')
    doc.print(f'Action space dim: {env.action_space.shape}')

    doc.print('Body masses:')
    doc.print(env.unwrapped.env.physics.named.model.body_mass)
    doc.print(f'Total mass: {np.sum(env.unwrapped.env.physics.named.model.body_mass)}')

    def print_feet_sliding_friction(env):
        doc.print('Feet sliding friction:')
        doc.print(f'bfoot: {env.unwrapped.env.physics.named.model.geom_friction["bfoot"][0]}')
        doc.print(f'ffoot: {env.unwrapped.env.physics.named.model.geom_friction["ffoot"][0]}')

    print_feet_sliding_friction(env)

with doc:
    """
    Capsule:

    cylinder + 2 half-spheres

    dimensions: radius (r), half-height of the cylinder (h)

    volume: 
    two half-spheres 4/3 pi r^3
    cylinder pi r^2 * 2 h

    total volume: pi (4/3 r^3 + r^2 * 2 h)
    """

    def get_capsule_volume(r, h):
        return np.pi * (4.0/3.0 * r**3 + r**2 * 2.0 * h)

    density = 1000
    torso_mass = density * get_capsule_volume(*env.unwrapped.env.physics.named.model.geom_size['torso'][:2])
    head_mass = density * get_capsule_volume(*env.unwrapped.env.physics.named.model.geom_size['head'][:2])
    doc.print('Torso mass:', torso_mass)
    doc.print('Head mass:', head_mass)
    doc.print('Torso + Head mass:', torso_mass + head_mass)
    doc.print('Torso to head ratio:', torso_mass / head_mass)

with doc:
    obses = []
    frames = []
    actions = []
    img_size = 256
    num_frames = 300
    seed = 1
    rng = np.random.default_rng(seed)

    obs = env.reset()
    obses.append(obs)
    for _ in range(num_frames):
        action = rng.normal(size=env.action_space.shape[0]) * 2.0
        actions.append(action)
        obs, reward, done, info = env.step(action)
        obses.append(obs)
        frames.append(env.render('rgb_array', width=img_size, height=img_size))

    print('Saving video...')
    frames = np.stack(frames)
    video_path = f'{Path(__file__).stem}/video_default.mp4'
    imageio.mimsave(video_path, img_as_ubyte(frames), format='mp4', fps=24)

    # with doc.table():
    #     row = doc.figure_row()

    doc.print('Default parameters (torso_mass=6.25)')
    doc.video(frames=None, src=video_path)

doc @ "## Varying torso mass"

with doc:
    obses = []
    frames = []
    obs = env.reset(torso_mass=100.0)
    doc.print('Body masses:')
    doc.print(env.unwrapped.env.physics.named.model.body_mass)
    doc.print(f'Total mass: {np.sum(env.unwrapped.env.physics.named.model.body_mass)}')

    obses.append(obs)
    for i in range(num_frames):
        obs, reward, done, info = env.step(actions[i])
        obses.append(obs)
        frames.append(env.render('rgb_array', width=img_size, height=img_size))

    print('Saving video...')
    frames = np.stack(frames)
    video_path = f'{Path(__file__).stem}/video_torso_mass_100.mp4'
    imageio.mimsave(video_path, img_as_ubyte(frames), format='mp4', fps=24)

    # with doc.table():
    #     row = doc.figure_row()

    doc.print('Very heavy torso (torso_mass=100.0)')
    doc.video(frames=None, src=video_path)

with doc:
    obses = []
    frames = []
    obs = env.reset(torso_mass=15.0)
    obses.append(obs)
    for i in range(num_frames):
        obs, reward, done, info = env.step(actions[i])
        obses.append(obs)
        frames.append(env.render('rgb_array', width=img_size, height=img_size))

    print('Saving video...')
    frames = np.stack(frames)
    video_path = f'{Path(__file__).stem}/video_torso_mass_15.mp4'
    imageio.mimsave(video_path, img_as_ubyte(frames), format='mp4', fps=24)

    # with doc.table():
    #     row = doc.figure_row()

    doc.print('Heavy torso (torso_mass=15.0)')
    doc.video(frames=None, src=video_path)

with doc:
    obses = []
    frames = []
    obs = env.reset(torso_mass=25.0)
    obses.append(obs)
    for i in range(num_frames):
        obs, reward, done, info = env.step(actions[i])
        obses.append(obs)
        frames.append(env.render('rgb_array', width=img_size, height=img_size))

    print('Saving video...')
    frames = np.stack(frames)
    video_path = f'{Path(__file__).stem}/video_torso_mass_25.mp4'
    imageio.mimsave(video_path, img_as_ubyte(frames), format='mp4', fps=24)

    # with doc.table():
    #     row = doc.figure_row()

    doc.print('Heavy torso (torso_mass=25.0)')
    doc.video(frames=None, src=video_path)

doc @ "## Varying foot Friction"

with doc:
    obses = []
    frames = []
    obs = env.reset(foot_sliding_friction=0.001)
    print_feet_sliding_friction(env)
    obses.append(obs)
    for i in range(num_frames):
        obs, reward, done, info = env.step(actions[i])
        obses.append(obs)
        frames.append(env.render('rgb_array', width=img_size, height=img_size))

    print('Saving video...')
    frames = np.stack(frames)
    video_path = f'{Path(__file__).stem}/video_foot_friction_0_001.mp4'
    imageio.mimsave(video_path, img_as_ubyte(frames), format='mp4', fps=24)

    # with doc.table():
    #     row = doc.figure_row()

    doc.print('foot_sliding_friction=0.001')
    doc.video(frames=None, src=video_path)

with doc:
    obses = []
    frames = []
    obs = env.reset(foot_sliding_friction=0.1)
    print_feet_sliding_friction(env)
    obses.append(obs)
    for i in range(num_frames):
        obs, reward, done, info = env.step(actions[i])
        obses.append(obs)
        frames.append(env.render('rgb_array', width=img_size, height=img_size))

    print('Saving video...')
    frames = np.stack(frames)
    video_path = f'{Path(__file__).stem}/video_foot_friction_0_1.mp4'
    imageio.mimsave(video_path, img_as_ubyte(frames), format='mp4', fps=24)

    # with doc.table():
    #     row = doc.figure_row()

    doc.print('foot_sliding_friction=0.1')
    doc.video(frames=None, src=video_path)

with doc:
    obses = []
    frames = []
    obs = env.reset(foot_sliding_friction=2.0)
    print_feet_sliding_friction(env)
    obses.append(obs)
    for i in range(num_frames):
        obs, reward, done, info = env.step(actions[i])
        obses.append(obs)
        frames.append(env.render('rgb_array', width=img_size, height=img_size))

    print('Saving video...')
    frames = np.stack(frames)
    video_path = f'{Path(__file__).stem}/video_foot_friction_2.mp4'
    imageio.mimsave(video_path, img_as_ubyte(frames), format='mp4', fps=24)

    # with doc.table():
    #     row = doc.figure_row()

    doc.print('foot_sliding_friction=2.0')
    doc.video(frames=None, src=video_path)

doc.flush()
