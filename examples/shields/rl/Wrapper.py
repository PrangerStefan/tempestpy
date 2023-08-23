import gymnasium as gym
import numpy as np


from gymnasium.spaces import Dict, Box
from collections import deque
from ray.rllib.utils.numpy import one_hot

class OneHotWrapper(gym.core.ObservationWrapper):
    def __init__(self, env, vector_index, framestack):
        super().__init__(env)
        self.framestack = framestack
        # 49=7x7 field of vision; 11=object types; 6=colors; 3=state types.
        # +4: Direction.
        self.single_frame_dim = 49 * (11 + 6 + 3) + 4
        self.init_x = None
        self.init_y = None
        self.x_positions = []
        self.y_positions = []
        self.x_y_delta_buffer = deque(maxlen=100)
        self.vector_index = vector_index
        self.frame_buffer = deque(maxlen=self.framestack)
        for _ in range(self.framestack):
            self.frame_buffer.append(np.zeros((self.single_frame_dim,)))

        self.observation_space = Dict(
            {
                "data": gym.spaces.Box(0.0, 1.0, shape=(self.single_frame_dim * self.framestack,), dtype=np.float32),
                "avail_actions": gym.spaces.Box(0, 10, shape=(10,), dtype=int),
            }
            ) 
        
        
        print(F"Set obersvation space to {self.observation_space}")
        

    def observation(self, obs):
        # Debug output: max-x/y positions to watch exploration progress.
        # print(F"Initial observation in Wrapper {obs}")
        if self.step_count == 0:
            for _ in range(self.framestack):
                self.frame_buffer.append(np.zeros((self.single_frame_dim,)))
            if self.vector_index == 0:
                if self.x_positions:
                    max_diff = max(
                        np.sqrt(
                            (np.array(self.x_positions) - self.init_x) ** 2
                            + (np.array(self.y_positions) - self.init_y) ** 2
                        )
                    )
                    self.x_y_delta_buffer.append(max_diff)
                    print(
                        "100-average dist travelled={}".format(
                            np.mean(self.x_y_delta_buffer)
                        )
                    )
                    self.x_positions = []
                    self.y_positions = []
                self.init_x = self.agent_pos[0]
                self.init_y = self.agent_pos[1]

      
        self.x_positions.append(self.agent_pos[0])
        self.y_positions.append(self.agent_pos[1])
        
        image = obs["data"]

        # One-hot the last dim into 11, 6, 3 one-hot vectors, then flatten.
        objects = one_hot(image[:, :, 0], depth=11)
        colors = one_hot(image[:, :, 1], depth=6)
        states = one_hot(image[:, :, 2], depth=3)
      
        all_ = np.concatenate([objects, colors, states], -1)
        all_flat = np.reshape(all_, (-1,))
        direction = one_hot(np.array(self.agent_dir), depth=4).astype(np.float32)
        single_frame = np.concatenate([all_flat, direction])
        self.frame_buffer.append(single_frame)
        
        #obs["one-hot"] = np.concatenate(self.frame_buffer)
        tmp = {"data": np.concatenate(self.frame_buffer), "avail_actions": obs["avail_actions"] }
        return tmp#np.concatenate(self.frame_buffer)


class MiniGridEnvWrapper(gym.core.Wrapper):
    def __init__(self, env):
        super(MiniGridEnvWrapper, self).__init__(env)
        self.observation_space = Dict(
            {
                "data": env.observation_space.spaces["image"],
                "avail_actions" : Box(0, 10, shape=(10,), dtype=np.int8),
            }
        )
        
        
    def test(self):
        print("Testing some stuff")
    
    def reset(self, *, seed=None, options=None):
        obs, infos = self.env.reset()
        return {
            "data": obs["image"],
            "avail_actions": np.array([0.0] * 10, dtype=np.int8)
        }, infos
    
    def step(self, action):
        orig_obs, rew, done, truncated, info = self.env.step(action)
        
        self.test()
        #print(F"Original observation is {orig_obs}")
        obs = {
            "data": orig_obs["image"],
            "avail_actions":  np.array([0.0] * 10, dtype=np.int8),
        }
        
        #print(F"Info is {info}")
        return obs, rew, done, truncated, info
    
    


class ImgObsWrapper(gym.core.ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission.

    Example:
        >>> import gymnasium as gym
        >>> from minigrid.wrappers import ImgObsWrapper
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> obs, _ = env.reset()
        >>> obs.keys()
        dict_keys(['image', 'direction', 'mission'])
        >>> env = ImgObsWrapper(env)
        >>> obs, _ = env.reset()
        >>> obs.shape
        (7, 7, 3)
    """

    def __init__(self, env):
        """A wrapper that makes image the only observation.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.observation_space = env.observation_space.spaces["image"]
        print(F"Set obersvation space to {self.observation_space}")

    def observation(self, obs):
        #print(F"obs in img obs wrapper {obs}")
        tmp = {"data": obs["image"], "Test": obs["Test"]}
        
        return tmp
