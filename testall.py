#!/usr/bin/python3

import subprocess

slippery_configs=["/home/knolli/Documents/University/Thesis/tempest-py/slippery_prob_075.yaml", 
                  "/home/knolli/Documents/University/Thesis/tempest-py/slippery_prob_08.yaml",
                  "/home/knolli/Documents/University/Thesis/tempest-py/slippery_prob_085.yaml",
                  "/home/knolli/Documents/University/Thesis/tempest-py/slippery_prob_09.yaml",
                  "/home/knolli/Documents/University/Thesis/tempest-py/slippery_prob_095.yaml",
                  "/home/knolli/Documents/University/Thesis/tempest-py/slippery_prob_1.yaml"]

slippery_probs=[[0.125, 0.25, 0.75], # 0.75
                [0.1, 0.2, 0.8],     # 0.8
                [0.075,0.15,0.85],   # 0.85
                [0.05, 0.1, 0.9],    # 0.9
                [0.025,0.05, 0.95],  # 0.95
                # [0.01, 0.02, 0.98],  # 0.98
                # [0.005,0.01, 0.99],  # 0.99
                [0, 0, 1]]           # 1

shield_values = [0.85, 0.9, 0.95, 0.98, 0.99, 1]

prob_confs = list(zip(slippery_probs, slippery_configs))
counter=1
shielding=["full", "none"]
comps= ["relative", "absolute"]

for sh in shielding:
  for shield_value in shield_values:
    for sh_comp in ["relative", "absolute"]:
      for probs, config in prob_confs:
        command = f"echo \"Running experiment with sh:{sh}, sh_value:{probs[2]}, sh_comp:{sh_comp}, probvalues:{probs}, config{config}\""    
        execute_command = f'./syncscript_local.sh {250000} {"../logresults/"} {"70"} {"MiniGrid-LavaSlipperyS12-v2"} {sh} {sh_comp} {config} {probs[0]} {probs[1]} {probs[2]} {shield_value}'
        subprocess.call(execute_command, shell=True)#.decode("utf-8").split('\n')
    