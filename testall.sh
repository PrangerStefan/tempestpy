#!/usr/bin/python3

import subprocess

slippery_configs=["slippery_prob_075.yaml", "slippery_prob_08.yaml","slippery_prob_085.yaml","slippery_prob_09.yaml",
                  "slippery_prob_095.yaml", "slippery_prob_098.yaml","slippery_prob_099.yaml","slippery_prob_1.yaml"]

slippery_probs=[[0.125, 0.25, 0.75], # 0.75
                [0.1, 0.2, 0.8],     # 0.8
                [0.075,0.15,0.85],   # 0.85
                [0.05, 0.1, 0.9],    # 0.9
                [0.025,0.05, 0.95],  # 0.95
                [0.01, 0.02, 0.98],  # 0.98
                [0.005,0.01, 0.99],  # 0.99
                [0, 0, 1]]           # 1

prob_confs = list(zip(slippery_probs, slippery_configs))
counter=1
shielding=["full", "none"]
comps= ["relative", "absolute"]

for sh in shielding:
  for sh_comp in ["relative", "absolute"]:
    for probs, config in prob_confs:
      command = f"echo \"Running experiment with sh:{sh}, sh_value:{probs[2]}, sh_comp:{sh_comp}, probvalues:{probs}, config{config}\""    
      execute_command = f'./syncscript.sh {250000} {"../logresults/"} {"70"} {"MiniGrid-LavaSlipperyS12-v2"} {sh} {sh_comp} {config} {probs[0]} {probs[1]} {probs[2]}'
      subprocess.call(execute_command, shell=True)#.decode("utf-8").split('\n')
  