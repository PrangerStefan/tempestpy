#!/usr/bin/python3

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

abs_path = os.getcwd()

slippery_configs=[f"{abs_path}/slippery_prob_075.yaml",
                  # f"{abs_path}/slippery_prob_08.yaml",
                  f"{abs_path}/slippery_prob_085.yaml",
                  # f"{abs_path}/slippery_prob_09.yaml",
                  f"{abs_path}/slippery_prob_095.yaml",
                  # f"{abs_path}/slippery_prob_1.yaml"
                  ]

slippery_probs=[[0.25, 0.75], # 0.75
                #[0.1, 0.2, 0.8],     # 0.8
                [0.15,0.85],   # 0.85
                #[0.05, 0.1, 0.9],    # 0.9
                [0.05, 0.95],  # 0.95
                # [0.01, 0.02, 0.98],  # 0.98
                # [0.005,0.01, 0.99],  # 0.99
                #[0, 0, 1]           # 1
                ]
prob_turn_intended = 1


#shield_values = [0.85, 0.9, 0.95, 0.98, 0.99, 1]
shield_values = [0.85, .95, 1]

prob_confs = list(zip(slippery_probs, slippery_configs))
counter = 1
shielding = ["full", "none"]
comparison_type = ["relative", "absolute"]
comps = ["relative", "absolute"]

NUM_TIMESTEPS=250000
LOGDIR="../logresults/"
ENV="MiniGrid-LavaSlipperyCliffS12-v0"

tasks = list()
NUM_WORKER=2

def run_command(command, logname):
  log = open(f"{logname}.log", "w")
  print(f"running {command}")
  subprocess.call(command, shell=True, stdout=log)#.decode("utf-8").split('\n')
  close(log)


# matrix for shielded runs
for shield_value in shield_values:
  for sh_comp in comparison_type:
    for probs, config in prob_confs:
      command = f"echo \"Running experiment with shielding full, sh_value:{shield_value}, sh_comp:{sh_comp}, probvalues:{probs}, config{config}\""
      execute_command = f'./syncscript.sh {NUM_TIMESTEPS} {LOGDIR} {"70"} {ENV} full {sh_comp} {config} {probs[0]} {probs[1]} {0} {shield_value} {prob_turn_intended}'
      print(execute_command)
      logname = f"shielded_comp_{sh_comp}_value_{shield_value}_probvalues{'_'.join(format(p, '10.3f') for p in probs)}_{config.split('/')[-1]}".replace(" ", "")
      tasks.append((execute_command, logname))

# loop for unshielded runs
for probs, config in prob_confs:
  command = f"echo \"Running experiment with shielding none, sh_value:0.0, sh_comp:{sh_comp}, probvalues:{probs}, config{config}\""
  execute_command = f'./syncscript.sh {NUM_TIMESTEPS} {LOGDIR} {"70"} {ENV} none {sh_comp} {config} {probs[0]} {probs[1]} {0} {shield_value} {prob_turn_intended}'
  logname = f"unshielded_probvalues{'_'.join(format(p, '10.3f') for p in probs)}_{config.split('/')[-1]}".replace(" ", "")
  print(execute_command)
  tasks.append((execute_command, logname))

print(f"Going to execute {len(tasks)} with {NUM_WORKER} threads... press Enter to start")
input("")

with ThreadPoolExecutor(max_workers=NUM_WORKER) as e:
    for task in tasks:
        print(f"submitted {task}")
        e.submit(run_command, task[0], task[1])
