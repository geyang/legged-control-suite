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


doc @ "# Friction test"

with doc:
    env = gym.make('lcs:Frictiontest-run-v1')
    obs = env.reset()


with doc:
    def get_pos(env):
        half_length = 0.5
        com_xyz = env.unwrapped.env.physics.named.data.xipos['torso']
        cos_phi = env.unwrapped.env.physics.named.data.xmat['torso', 'xz']
        sin_phi = env.unwrapped.env.physics.named.data.xmat['torso', 'zz']
        x_right = com_xyz[0] + half_length * sin_phi
        return x_right

with doc:
    table = doc.table()
    img_size = 256
    num_frames = 300

    row = None
    for i, sliding_friction in enumerate([0.001, 0.1, None, 2.0]):
        if i % 2 == 0:
            row = table.figure_row()
        positions = []
        frames = []

        if sliding_friction is not None:
            obs = env.reset(sliding_friction=sliding_friction)
        else:
            obs = env.reset()

        doc.print(f'Mujoco model frictions {env.unwrapped.env.physics.named.model.geom_friction["torso"]}')
        positions.append(get_pos(env))
        for _ in range(num_frames):
            action = np.zeros(env.action_space.shape)
            obs, reward, done, info = env.step(action)
            positions.append(get_pos(env))
            frames.append(env.render('rgb_array', width=img_size, height=img_size))

        frames = np.stack(frames)
        friction_str = 'default (1.0)' if sliding_friction is None else f'{sliding_friction:0.3f}'
        friction_fstr = 'default_1' if sliding_friction is None else str(sliding_friction).replace('.', '_')
        video_path = f'{Path(__file__).stem}/video_{friction_fstr}.mp4'
        imageio.mimsave(video_path, img_as_ubyte(frames), format='mp4', fps=24)


        plt.plot(positions, lw=3)
        plt.ylim(-0.55, 0.55)
        plt.axhline(0.0, color='black', linestyle='--')
        plt.xlabel('Steps', fontsize=16)
        plt.ylabel('Pole right end position', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        row.savefig(f'{Path(__file__).stem}/plot_{friction_fstr}.png', title=f'sliding friction: {friction_str}')


for sliding_friction in [0.001, 0.1, None, 2.0]:
    friction_str = 'default (1.0)' if sliding_friction is None else f'{sliding_friction:0.3f}'
    doc.print(f'sliding friction: {friction_str}')
    friction_fstr = 'default_1' if sliding_friction is None else str(sliding_friction).replace('.', '_')
    video_path = f'{Path(__file__).stem}/video_{friction_fstr}.mp4'

    doc.video(frames=None, src=video_path)

doc.flush()
