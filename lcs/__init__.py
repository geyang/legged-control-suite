import collections
import inspect

from lcs import bipedalwalker
from lcs import paramcartpole
from lcs import paramcheetah
from lcs import frictiontest

# Find all domains imported.
_DOMAINS = {name: module for name, module in locals().items()
            if inspect.ismodule(module) and hasattr(module, 'SUITE')}


def _get_tasks(tag):
  """Returns a sequence of (domain name, task name) pairs for the given tag."""
  result = []

  for domain_name in sorted(_DOMAINS.keys()):

    domain = _DOMAINS[domain_name]

    if tag is None:
      tasks_in_domain = domain.SUITE
    else:
      tasks_in_domain = domain.SUITE.tagged(tag)

    for task_name in tasks_in_domain.keys():
      result.append((domain_name, task_name))

  return tuple(result)


def _get_tasks_by_domain(tasks):
  """Returns a dict mapping from task name to a tuple of domain names."""
  result = collections.defaultdict(list)

  for domain_name, task_name in tasks:
    result[domain_name].append(task_name)

  return {k: tuple(v) for k, v in result.items()}


# A sequence containing all (domain name, task name) pairs.
ALL_TASKS = _get_tasks(tag=None)

# Subsets of ALL_TASKS, generated via the tag mechanism.
BENCHMARKING = _get_tasks('benchmarking')
EASY = _get_tasks('easy')
HARD = _get_tasks('hard')
EXTRA = tuple(sorted(set(ALL_TASKS) - set(BENCHMARKING)))
NO_REWARD_VIZ = _get_tasks('no_reward_visualization')
REWARD_VIZ = tuple(sorted(set(ALL_TASKS) - set(NO_REWARD_VIZ)))

# A mapping from each domain name to a sequence of its task names.
TASKS_BY_DOMAIN = _get_tasks_by_domain(ALL_TASKS)


def load(domain_name, task_name, task_kwargs=None, environment_kwargs=None,
         visualize_reward=False):
  """Returns an environment from a domain name, task name and optional settings.

  ```python
  env = suite.load('cartpole', 'balance')
  ```

  Args:
    domain_name: A string containing the name of a domain.
    task_name: A string containing the name of a task.
    task_kwargs: Optional `dict` of keyword arguments for the task.
    environment_kwargs: Optional `dict` specifying keyword arguments for the
      environment.
    visualize_reward: Optional `bool`. If `True`, object colours in rendered
      frames are set to indicate the reward at each step. Default `False`.

  Returns:
    The requested environment.
  """
  return build_environment(domain_name, task_name, task_kwargs,
                           environment_kwargs, visualize_reward)


def build_environment(domain_name, task_name, task_kwargs=None,
                      environment_kwargs=None, visualize_reward=False):
  """Returns an environment from the suite given a domain name and a task name.

  Args:
    domain_name: A string containing the name of a domain.
    task_name: A string containing the name of a task.
    task_kwargs: Optional `dict` specifying keyword arguments for the task.
    environment_kwargs: Optional `dict` specifying keyword arguments for the
      environment.
    visualize_reward: Optional `bool`. If `True`, object colours in rendered
      frames are set to indicate the reward at each step. Default `False`.

  Raises:
    ValueError: If the domain or task doesn't exist.

  Returns:
    An instance of the requested environment.
  """
  if domain_name not in _DOMAINS:
    raise ValueError('Domain {!r} does not exist.'.format(domain_name))

  domain = _DOMAINS[domain_name]

  if task_name not in domain.SUITE:
    raise ValueError('Level {!r} does not exist in domain {!r}.'.format(
        task_name, domain_name))

  task_kwargs = task_kwargs or {}
  if environment_kwargs is not None:
    task_kwargs = dict(task_kwargs, environment_kwargs=environment_kwargs)
  env = domain.SUITE[task_name](**task_kwargs)
  env.task.visualize_reward = visualize_reward
  return env


import numpy as np
from gym import spaces
from gym.envs import register
from gym.envs.registration import EnvSpec
from gym_dmc.dmc_env import DMCEnv, convert_dm_control_to_gym_space


class LCSEnv(DMCEnv):
    def __init__(self, domain_name, task_name,
                 task_kwargs=None,
                 environment_kwargs=None,
                 visualize_reward=False,
                 height=84,
                 width=84,
                 camera_id=0,
                 frame_skip=1,
                 channels_first=True,
                 from_pixels=False,
                 gray_scale=False,
                 warmstart=True,  # info: https://github.com/deepmind/dm_control/issues/64
                 no_gravity=False,
                 non_newtonian=False,
                 skip_start=None,  # useful in Manipulator for letting things settle
                 space_dtype=None,  # default to float for consistency
                 ):
        self.env = load(domain_name,
                        task_name,
                        task_kwargs=task_kwargs,
                        environment_kwargs=environment_kwargs,
                        visualize_reward=visualize_reward)
        self.metadata = {'render.modes': ['human', 'rgb_array'],
                         'video.frames_per_second': round(1.0 / self.env.control_timestep())}

        self.from_pixels = from_pixels
        self.gray_scale = gray_scale
        self.channels_first = channels_first
        obs_spec = self.env.observation_spec()
        if from_pixels:
            color_dim = 1 if gray_scale else 3
            image_shape = [color_dim, width, height] if channels_first else [width, height, color_dim]
            self.observation_space = convert_dm_control_to_gym_space(
                obs_spec, dtype=space_dtype,
                pixels=spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8)
            )
        else:
            self.observation_space = convert_dm_control_to_gym_space(obs_spec, dtype=space_dtype)
        self.action_space = convert_dm_control_to_gym_space(self.env.action_spec(), dtype=space_dtype)
        self.viewer = None

        self.render_kwargs = dict(
            height=height,
            width=width,
            camera_id=camera_id,
        )
        self.frame_skip = frame_skip
        if not warmstart:
            self.env.physics.data.qacc_warmstart[:] = 0
        self.no_gravity = no_gravity
        self.non_newtonian = non_newtonian

        if self.no_gravity:  # info: this removes gravity.
            self.turn_off_gravity()

        self.skip_start = skip_start

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs).observation
        for i in range(self.skip_start or 0):
            obs = self.env.step([0]).observation

        if self.from_pixels:
            obs['pixels'] = self._get_obs_pixels()

        return obs


def make_gym_env(flatten_obs=True, from_pixels=False, frame_skip=1, episode_frames=1000, id=None, **kwargs):
    max_episode_steps = episode_frames / frame_skip

    env = LCSEnv(from_pixels=from_pixels, frame_skip=frame_skip, **kwargs)

    # This spec object gets picked up by the gym.EnvSpecs constructor
    # used in gym.registration.EnvSpec.make, L:93 to generate the spec
    if id:
        env._spec = EnvSpec(id=f"{domain_name.capitalize()}-{task_name}-v1", max_episode_steps=max_episode_steps)

    if from_pixels:
        from gym_dmc.wrappers import ObservationByKey
        env = ObservationByKey(env, "pixels")
    elif flatten_obs:
        from gym_dmc.wrappers import FlattenObservation
        env = FlattenObservation(env)
    return env


for domain_name, task_name in ALL_TASKS:
    ID = f'{domain_name.capitalize()}-{task_name}-v1'
    register(id=ID,
             entry_point='lcs:make_gym_env',
             kwargs=dict(
                 id=ID,
                 domain_name=domain_name,
                 task_name=task_name,
                 channels_first=True,
                 width=84,
                 height=84,
                 frame_skip=1),
             )
