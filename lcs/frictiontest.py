# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Pole Domain."""

import collections
from pathlib import Path

import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards


# How long the simulation will run, in seconds.
_DEFAULT_TIME_LIMIT = 10

# Running speed above which reward is 1.
_RUN_SPEED = 10

SUITE = containers.TaggedTasks()


class ParametricEnvironment(control.Environment):
    def reset(self, **kwargs):
        new_xml = _make_model(**kwargs)
        self.physics.reload_from_xml_string(new_xml, common.ASSETS)
        return super().reset()

    def change_model(self, **kwargs):
        qpos = self.physics.data.qpos.copy()
        qvel = self.physics.data.qvel.copy()

        new_xml = _make_model(**kwargs)
        self.physics.reload_from_xml_string(new_xml, common.ASSETS)

        with self.physics.reset_context():
            self.physics.data.qpos[:] = qpos
            self.physics.data.qvel[:] = qvel

        self.task.after_step(self.physics)


def _make_model(sliding_friction=1.0):

    with open(Path(__file__).with_suffix('.xml')) as f:
        xml_string = f.read()

    # We need to set priority to 1 to directly set the friction coefficient
    # Otherwise mujoco will use the max(ground_friction=1.0, pole_friction)


    xml_string = xml_string.format(
        sliding_friction=sliding_friction,
    )

    return xml_string



def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return _make_model(), common.ASSETS


@SUITE.add('benchmarking')
def run(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the run task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Pole(random=random)
  environment_kwargs = environment_kwargs or {}
  return ParametricEnvironment(physics, task, time_limit=time_limit,
                               **environment_kwargs)


class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Pole domain."""

  def speed(self):
    """Returns the horizontal speed of the Pole."""
    return self.named.data.sensordata['torso_subtreelinvel'][0]


class Pole(base.Task):
  """A `Task` to train a running Pole."""

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode."""
    # The indexing below assumes that all joints have a single DOF.
    # assert physics.model.nq == physics.model.njnt
    # is_limited = physics.model.jnt_limited == 1
    # lower, upper = physics.model.jnt_range[is_limited].T
    # physics.data.qpos[is_limited] = self.random.uniform(lower, upper)
    #
    # # Stabilize the model before the actual simulation.
    # physics.step(nstep=1)
    #
    # physics.data.time = 0
    # self._timeout_progress = 0

    physics.named.data.qpos['rooty'] = np.pi * 0.48
    super().initialize_episode(physics)

  def get_observation(self, physics):
    """Returns an observation of the state, ignoring horizontal position."""
    obs = collections.OrderedDict()
    # Ignores horizontal position to maintain translational invariance.
    obs['position'] = physics.data.qpos[1:].copy()
    obs['velocity'] = physics.velocity()
    return obs

  def get_reward(self, physics):
    """Returns a reward to the agent."""
    return rewards.tolerance(physics.speed(),
                             bounds=(_RUN_SPEED, float('inf')),
                             margin=_RUN_SPEED,
                             value_at_margin=0,
                             sigmoid='linear')
