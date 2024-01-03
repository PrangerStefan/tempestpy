#!/bin/bash

. ../env/bin/activate

commit_message="$(git log -1 --pretty=format:"%s_%h" | sed 's/\s/_/g')"
git diff --quiet || commit_message="${commit_message}_dirty"
datetime="$(date +%Y%-m-%dT%H:%M:%S)"


#cpu=nehalem192g0
cpu=epyc512g0
gpu=i9-64g0
#cpu=haswell378g0

env=$4
shielding=$5
num_steps=$1
experiment_log_dir="${2}"
num_evaluations=$3
shield_value=$6
prism_config=$7
prob_displacement=$8
prob_intended=$9
prob_turn_displacement="${10}"
prop_turn_intended="${12}"
shield_comparision="${11}"
NUM_GPUS="1"

exp_name="${commit_message}-${datetime}-env:${env}-sh:${shielding}-value:${shield_value}-comp:${shield_comparision}-prob:${prob_intended}"
experiment_log_dir="${2}/${exp_name}"


MINIGRID_BINARY=""
if [ "$(whoami)" = "spranger" ]; then
	MINIGRID_BINARY="/workstore/spranger/tempestpy/Minigrid2PRISM/build/main"
else
	MINIGRID_BINARY="/workstore/tknoll/Minigrid2PRISM/build/main"
fi

# echo $experiment_log_dir
# echo $(pwd)
# echo $(pwd)/$experiment_log_dir/$exp_name
# python3 examples/shields/rl/11_minigridrl.py --expname "$exp_name" --steps "$1" --log_dir "$experiment_log_dir"/ --evaluations "$3" --env "$4" --shielding "$5" --shield_comparision "$6" --prism_config "$7" --prob_next "$8" --prob_direct "$9" --prob_forward "${10}"  --shield_value "${11}" &

set -x
srun -w $gpu python3 examples/shields/rl/15_train_eval_tune.py \
     --expname "$exp_name" \
     --log_dir "$experiment_log_dir" \
     --grid_to_prism_binary_path $MINIGRID_BINARY \
     --steps $1 \
     --evaluations $3 \
     --env $4 \
     --shielding $5 \
     --shield_comparision $6 \
     --prism_config $7  \
     --prob_displacement $8 \
     --prob_intended $9 \
     --prob_turn_displacement "${10}" \
     --shield_value "${11}" \
     --prop_turn_intended "${12}" \
     --num_gpus ${NUM_GPUS} &
set +x

sleep 20
rsync -avtr --stats $(pwd)/$experiment_log_dir/$exp_name tensorboard:/media/data1/easy_rl_tb_logs
sleep 60
while [[ -n $(jobs -r) ]]; do
  rsync -avtr --append --stats $(pwd)/$experiment_log_dir/$exp_name tensorboard:/media/data1/easy_rl_tb_logs
  sleep 60;
done
rsync -avtr --append --stats $(pwd)/$experiment_log_dir/$exp_name tensorboard:/media/data1/easy_rl_tb_logs
