
from typing import Dict

from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2

from ray.rllib.algorithms.callbacks import DefaultCallbacks, make_multi_callbacks

class MyCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[PolicyID, Policy], episode: Episode | EpisodeV2, env_index: int | None = None, **kwargs) -> None:
        # print(F"Epsiode started Environment: {base_env.get_sub_environments()}")
        env = base_env.get_sub_environments()[0]
        episode.user_data["count"] = 0
        episode.user_data["ran_into_lava"] = []
        episode.user_data["goals_reached"] = []
        episode.hist_data["ran_into_lava"] = []
        episode.hist_data["goals_reached"] = []
        # print("On episode start print")
        # print(env.printGrid())
        # print(worker)
        # print(env.action_space.n)
        # print(env.actions)
        # print(env.mission)
        # print(env.observation_space)
        # img = env.get_frame()
        # plt.imshow(img)
        # plt.show()
    
       
    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[PolicyID, Policy] | None = None, episode: Episode | EpisodeV2, env_index: int | None = None, **kwargs) -> None:
         episode.user_data["count"] = episode.user_data["count"] + 1
         env = base_env.get_sub_environments()[0]
        #  print(env.printGrid())
    
    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[PolicyID, Policy], episode: Episode | EpisodeV2 | Exception, env_index: int | None = None, **kwargs) -> None:
        # print(F"Epsiode end Environment: {base_env.get_sub_environments()}")
        env = base_env.get_sub_environments()[0]
        agent_tile = env.grid.get(env.agent_pos[0], env.agent_pos[1])
    
        episode.user_data["goals_reached"].append(agent_tile is not None and agent_tile.type == "goal")
        episode.user_data["ran_into_lava"].append(agent_tile is not None and agent_tile.type == "lava")
        episode.custom_metrics["reached_goal"] = agent_tile is not None and agent_tile.type == "goal"
        episode.custom_metrics["ran_into_lava"] =  agent_tile is not None and agent_tile.type == "lava"
        #print("On episode end print")
        #print(env.printGrid())
        episode.hist_data["goals_reached"] = episode.user_data["goals_reached"]
        episode.hist_data["ran_into_lava"] = episode.user_data["ran_into_lava"]
    
        
    def on_evaluate_start(self, *, algorithm: Algorithm, **kwargs) -> None:
        print("Evaluate Start")
        
    def on_evaluate_end(self, *, algorithm: Algorithm, evaluation_metrics: dict, **kwargs) -> None:
        print("Evaluate End")
        
