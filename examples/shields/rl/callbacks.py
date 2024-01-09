
from typing import Dict, Optional
from ray.rllib.env.env_context import EnvContext

from ray.rllib.policy import Policy
from ray.rllib.utils.typing import EnvType, PolicyID

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2

from ray.rllib.algorithms.callbacks import DefaultCallbacks, make_multi_callbacks
from ray.tune import Callback

import matplotlib.pyplot as plt
    

class CustomCallback(DefaultCallbacks):
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[PolicyID, Policy], episode, env_index, **kwargs) -> None:
        env = base_env.get_sub_environments()[0]
        episode.user_data["count"] = 0
        episode.user_data["ran_into_lava"] = []
        episode.user_data["goals_reached"] = []
        episode.user_data["ran_into_adversary"] = []
        episode.hist_data["ran_into_lava"] = []
        episode.hist_data["goals_reached"] = []
        episode.hist_data["ran_into_adversary"] = []

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv, policies, episode, env_index, **kwargs) -> None:
        episode.user_data["count"] = episode.user_data["count"] + 1
        env = base_env.get_sub_environments()[0]
        # print(env.printGrid())


    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv, policies, episode, env_index, **kwargs) -> None:
        # print(F"Epsiode end Environment: {base_env.get_sub_environments()}")
        env = base_env.get_sub_environments()[0]
        agent_tile = env.grid.get(env.agent_pos[0], env.agent_pos[1])
        ran_into_adversary = False

        if hasattr(env, "adversaries") and env.adversaries:
            adversaries = env.adversaries.values()
            for adversary in adversaries:
                if adversary.cur_pos[0] == env.agent_pos[0] and adversary.cur_pos[1] == env.agent_pos[1]:
                    ran_into_adversary = True
                    break

        episode.user_data["goals_reached"].append(agent_tile is not None and agent_tile.type == "goal")
        episode.user_data["ran_into_lava"].append(agent_tile is not None and agent_tile.type == "lava")
        episode.user_data["ran_into_adversary"].append(ran_into_adversary)
        episode.custom_metrics["reached_goal"] = agent_tile is not None and agent_tile.type == "goal"
        episode.custom_metrics["ran_into_lava"] =  agent_tile is not None and agent_tile.type == "lava"
        episode.custom_metrics["ran_into_adversary"] = ran_into_adversary
        #print("On episode end print")
        # print(env.printGrid())
        episode.hist_data["goals_reached"] = episode.user_data["goals_reached"]
        episode.hist_data["ran_into_lava"] = episode.user_data["ran_into_lava"]
        episode.hist_data["ran_into_adversary"] = episode.user_data["ran_into_adversary"]

    def on_evaluate_start(self, *, algorithm: Algorithm, **kwargs) -> None:
        print("Evaluate Start")

    def on_evaluate_end(self, *, algorithm: Algorithm, evaluation_metrics: dict, **kwargs) -> None:
        print("Evaluate End")
