from typing import Dict, Optional, Union

from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN, FLOAT_MAX

torch, nn = try_import_torch()



class TorchActionMaskModel(TorchModelV2, nn.Module):
    """PyTorch version of above ActionMaskingModel."""

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
        custom_config = model_config['custom_model_config']
        print(F"Original Space is: {orig_space}")
        #print(model_config)
        print(F"Observation space in model: {obs_space}")
        print(F"Provided action space in model {action_space}")
        
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)
        
        assert("shield" in custom_config)
        
        self.shield = custom_config["shield"]
        self.count = 0

        self.internal_model = TorchFC(
            orig_space["data"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

        # disable action masking --> will likely lead to invalid actions
        self.no_masking = False
        if "no_masking" in model_config["custom_model_config"]:
            self.no_masking = model_config["custom_model_config"]["no_masking"]

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        # print(F"Input dict is {input_dict} at obs: {input_dict['obs']}")
        # print(F"State is {state}")        
        
        # print(input_dict["env"])

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["data"]})

      #  print(F"Caluclated Logits {logits} with size {logits.size()} Count: {self.count}")

        action_mask = input_dict["obs"]["avail_actions"]
        #print(F"Action mask is {action_mask} with dimension {action_mask.size()}")

        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        # assert(False)
        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        print(F"Infinity mask {inf_mask}, Masked logits {masked_logits}")

        # # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()
