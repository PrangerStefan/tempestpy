from typing import Dict, Optional, Union
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN, FLOAT_MAX

torch, nn = try_import_torch()



class TorchActionMaskModel(TorchModelV2, nn.Module):

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)
        
        self.count = 0

        self.internal_model = TorchFC(
            orig_space["data"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )
        

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["data"]})
   
        action_mask = input_dict["obs"]["action_mask"]

        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()
