import gymnasium as gym
import numpy as np 

class EpisodeLifeEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__()
        self.lives = 0
        self.done = True
        self.env = env 
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step()
        self.done = np.logical_or(terminated, truncated)
        lives = self.env.unwrapped.env._life
        if self.lives > lives and lives > 0:
            terminated, truncated = True, True
        self.lives = lives 

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.done:
            obs = self.env.reset()
        else:
            obs, _, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.env._life
        return obs
    
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        self.env = env

    def step(self, action):

        total_reward = 0.0
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step()
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
        # super().__init__() #
        self.env = env
        self.observation_space = gym.spaces.Box(
            low = 0,
            high = 255,
            shape = env.observation_space.shape,
            dtype = env.observation_space.shape,
        )
        self.action_space = gym.spaces.Discrete(
            env.action_space.n
        )
    
    def step(self, action):
        return self.env.step(action)
    
    def reset(self):
        return self.env.reset()
    
    def render(self):
        return self.env.render()
    
    def seed(self, seed=None):
        return self.env.seed(seed=seed)

class ImageTranspose(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(self, env)
        self.env = env
        shape = env.observation_space.shape
        dtype = env.observation_space.dtype
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(shape[2], shape[0], shape[1]),
            dtype=dtype,
        )
    def observation(self, observation):
        observation = np.transpose(observation, axes=[2, 0, 1])
        return observation
