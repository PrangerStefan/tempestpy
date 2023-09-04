import gymnasium as gym
import numpy as np

from minigrid.core.actions import Actions

from gymnasium.spaces import Dict, Box
from collections import deque
from ray.rllib.utils.numpy import one_hot

from helpers import get_action_index_mapping, create_shield_dict, extract_keys


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
                "action_mask": gym.spaces.Box(0, 10, shape=(env.action_space.n,), dtype=int),
            }
            )

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

        tmp = {"data": np.concatenate(self.frame_buffer), "action_mask": obs["action_mask"] }
        return tmp


class MiniGridEnvWrapper(gym.core.Wrapper):
    def __init__(self, env, args=None):
        super(MiniGridEnvWrapper, self).__init__(env)
        self.max_available_actions = env.action_space.n
        self.observation_space = Dict(
            {
                "data": env.observation_space.spaces["image"],
                "action_mask" : Box(0, 10, shape=(self.max_available_actions,), dtype=np.int8),
            }
        )
        self.args = args

    def create_action_mask(self):
        coordinates = self.env.agent_pos
        view_direction = self.env.agent_dir

        key_text = ""

        # only support one key for now
        if self.keys:
            key_text = F"!Agent_has_{self.keys[0]}_key\t& "


        if self.env.carrying and self.env.carrying.type == "key":
            key_text = F"Agent_has_{self.env.carrying.color}_key\t& "

        cur_pos_str = f"[{key_text}!AgentDone\t& xAgent={coordinates[0]}\t& yAgent={coordinates[1]}\t& viewAgent={view_direction}]"

        allowed_actions = []

      
        # Create the mask
        # If shield restricts action mask only valid with 1.0
        # else set all actions as valid
        mask = np.array([0.0] * self.max_available_actions, dtype=np.int8)

        if cur_pos_str in self.shield and self.shield[cur_pos_str]:
             allowed_actions = self.shield[cur_pos_str]
             for allowed_action in allowed_actions:
                 index =  get_action_index_mapping(allowed_action[1]) # Allowed_action is a set
                 if index is None:
                     assert(False)
                 mask[index] = 1.0
        else:
            for index, x in enumerate(mask):
                mask[index] = 1.0
        
        front_tile = self.env.grid.get(self.env.front_pos[0], self.env.front_pos[1])

        if front_tile is not None and front_tile.type == "key":
            mask[Actions.pickup] = 1.0
            
        # if self.env.carrying:
        #     mask[Actions.drop] = 1.0
            
        if front_tile and front_tile.type == "door":
            mask[Actions.toggle] = 1.0
        
        return mask

    def reset(self, *, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        self.shield = create_shield_dict(self.env, self.args)
        self.keys = extract_keys(self.env)
        mask = self.create_action_mask()
        return {
            "data": obs["image"],
            "action_mask": mask
        }, infos

    def step(self, action):
        orig_obs, rew, done, truncated, info = self.env.step(action)

        mask = self.create_action_mask()
        obs = {
            "data": orig_obs["image"],
            "action_mask": mask,
        }

        #print(F"Info is {info}")
        return obs, rew, done, truncated, info


