from typing import Callable, Dict

import gymnasium as gym
import gym as gym_old
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics

from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
# env = gym.make("MontezumaRevenge-v4", render_mode="rgb_array")
import numpy as np
import time

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game
        over.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True
        self.env = env

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = np.logical_or(terminated, truncated)
        lives = self.env.unwrapped.env._life
        if self.lives > lives > 0:
            terminated, truncated = True, True
        self.lives = lives
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.env._life
        return obs
    
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip
        self.env = env

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if np.logical_or(terminated, truncated):
                break
        return obs, total_reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        return self.env.reset()
    
    def render(self):
        return self.env.render()


class Gym2Gymnasium(gym.Wrapper):
    def __init__(self, env):
        """Convert gym.Env to gymnasium.Env"""
        self.env = env

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=env.observation_space.shape,
            dtype=env.observation_space.dtype,
        )
        self.action_space = gym.spaces.Discrete(env.action_space.n)

    def step(self, action):
        """Repeat action, and sum reward"""
        return self.env.step(action)

    def reset(self, options=None, seed=None):
        return self.env.reset()

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed=seed)
    
class ImageTranspose(gym.ObservationWrapper):
    """Transpose observation from channels last to channels first.

    Args:
        env (gym.Env): Environment to wrap.

    Returns:
        Minigrid2Image instance.
    """

    def __init__(self, env: gym.Env) -> None:
        gym.ObservationWrapper.__init__(self, env)
        shape = env.observation_space.shape
        dtype = env.observation_space.dtype
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(shape[2], shape[0], shape[1]),
            dtype=dtype,
        )

    def observation(self, observation):
        """Convert observation to image."""
        observation= np.transpose(observation, axes=[2, 0, 1])
        return observation

def make_mario_env(
        env_id: str = "SuperMarioBros-1-1-v3",
        num_envs: int = 8,
        device: str = "cpu",
        asynchronous: bool = True,
        seed: int = 0,
        gray_scale: bool = False,
        frame_stack: int = 0,
    ) :

    def make_env(env_id: str, seed: int) -> Callable:
        def _thunk():
            env = gym_old.make(env_id, apply_api_compatibility=True, render_mode="rgb_array")
            env = JoypadSpace(env, SIMPLE_MOVEMENT)
            env = Gym2Gymnasium(env)
            env = SkipFrame(env, skip=4)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            if gray_scale:
                env = gym.wrappers.GrayScaleObservation(env)
            if frame_stack > 0:
                env = gym.wrappers.FrameStack(env, frame_stack)
            if not gray_scale and frame_stack <= 0:
                env = ImageTranspose(env)
            env = EpisodicLifeEnv(env)
            env = gym.wrappers.TransformReward(env, lambda r: 0.01*r)
            env.observation_space.seed(seed)
            return env
        return _thunk
    
    envs = [make_env(env_id, seed + i) for i in range(num_envs)]
    if asynchronous:
        envs = AsyncVectorEnv(envs)
    else:
        envs = SyncVectorEnv(envs)
    
    envs = RecordEpisodeStatistics(envs)
    return envs

# def make_mario_env():
#     def thunk():
#         env = gym_super_mario_bros.make('SuperMarioBros-v0')
#         env = JoypadSpace(env, SIMPLE_MOVEMENT)
#         # env = Gym2Gymnasium(env)
#         return env
#     return thunk

def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        
        return env

    return thunk

if __name__ == '__main__':
    environment = 'Mario'
    if environment == 'Atari':
        envs = gym.vector.SyncVectorEnv(
                [make_env("ALE/MontezumaRevenge-v5", i, False, "Test") for i in range(2)],
            )
        # model = A2C("MlpPolicy", env, verbose=1)
        # model.learn(total_timesteps=10_000)
        
        obs, info = envs.reset()
        for i in range(10):
            # print(envs.action_space.sample())
            obs, reward, terminated, truncated, info = envs.step(envs.action_space.sample())
            # env.render()
            # VecEnv resets automatically
            print(obs.shape)
            print(envs.action_space)
            for env in range(2):
                if terminated[env] or truncated[env]:
                    observation, info = envs.reset()

        envs.close()
    else:
        # envs = gym_super_mario_bros.make('SuperMarioBros-v0')
        # envs = JoypadSpace(envs, SIMPLE_MOVEMENT)
        # # print(envs.action_space)
        # envs = Gym2Gymnasium(envs)
        envs = make_mario_env(num_envs=2)

        obs = envs.reset()
        # print(type(obs))
        for i in range(1000):
            print(envs.action_space.sample())
            obs, reward, terminated, truncated, info = envs.step(envs.action_space.sample())
            # envs.render()
            print(obs.shape)
            # VecEnv resets automatically
            print(envs.action_space)
            # time.sleep(0.1)
        envs.close()