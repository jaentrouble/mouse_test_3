import gym
from gym.spaces import Discrete, Dict, Box
from gym.utils import seeding
from .assets.engine import Engine
from .constants import rng
from .constants import engine_const as ec
import numpy as np

class MouseEnv(gym.Env) :
    metadata = {
        'render.modes' : ['human','rgb']
    }
    def __init__(self):
        #Turn left 45°, Move forward, Turn right 45°
        self.action_space = Discrete(3)
        self._done = False
        self.viewer = None
        self.engine = None
        self.max_step = 100
        self.cur_step = 0
        self.image_size = (720,720)
        self.seed()

        # 3 Continuous Inputs from both eyes
        self.observation_space = Dict(
            {'Right' : Box(0, 255, shape=(100,ec.CacheNum,3), dtype=np.uint8),
             'Left' : Box(0,255, shape=(100,ec.CacheNum,3), dtype = np.uint8)}
        )
        

    def step(self, action):
        assert not (self.engine is None), 'Reset first before starting env'
        if self._done :
            print('The game is already done. Continuing may cause unexpected'\
                ' behaviors')
        if action == 0:
            trans_action = ((0,0),np.pi/4)
        elif action == 1:
            trans_action = ((10,0),0)
        elif action == 2:
            trans_action = ((0,0),-np.pi/4)
        observation, reward, done, info = self.engine.update(trans_action)
        if done:
            self._done = True
        
        #Check if reached max_step
        self.cur_step += 1
        if self.cur_step >= self.max_step:
            self._done = True
            done = True

        return observation, reward, done, info

    def reset(self):
        """
        Reset the environment and return initial observation
        """
        self._done = False
        self.cur_step = 0
        self.engine = self._new_engine()
        initial_observation = self.engine.initial_observation()
        return initial_observation

    def render(self, mode='human'):
        assert not (self.engine is None), 'Reset first before starting env'
        if 'human' in mode :
            from gym.envs.classic_control import rendering
            if self.viewer == None:
                self.viewer = rendering.SimpleImageViewer(maxwidth=720)
            self.viewer.imshow(self.engine.image)
        elif 'rgb' in mode :
            return self.engine.image

    def seed(self, seed=None):
        np_random, seed = seeding.np_random(seed)
        rng.np_random = np_random
        self.action_space.seed(seed)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _new_engine(self):
        return Engine(*self.image_size)

# Testing
if __name__ == '__main__' :
    env = MouseEnv()
    env.render()
    a = input()