#!/bin/bash

for i in 0 1 2 3 4; do
    python3 train_agent.py --agent-type DQN --npk.random-reset --npk.domain-rand --npk.intvn-interval 7 --DQN.wandb-project-name sunflower_single_cost --agro-file sunflower_agro.yaml --track --save-folder data/Sunflower_Single_Cost/"$i"/ --env-reward RewardFertilizationCostWrapper --cost 2 --config-fpath data/Sunflower_Multi_Cost/DQN/multi-lnpkw-v0__rl_utils__1__1739474579/config_farm_"$i".yaml
done