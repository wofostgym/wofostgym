#!/bin/bash

python3 train_agent.py --agent-type PPO --PPO.wandb-project-name jujube_threshold_wk_rand --agro-file jujube_agro.yaml --env-id perennial-lnpkw-v0 --npk.random-reset --npk.domain-rand --npk.scale 0.03 --track --npk.intvn-interval 14 --save-folder data/Jujube_Threshold_WK_Rand/ --env-reward RewardFertilizationThresholdWrapper --max-n 80 --max-p 80 --max-k 80 --max-w 40
