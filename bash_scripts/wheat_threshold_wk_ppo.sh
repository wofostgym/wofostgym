#!/bin/bash

python3 train_agent.py --agent-type PPO --PPO.wandb-project-name threshold_wk_rand --npk.random-reset --npk.domain-rand --npk.scale 0.03 --track --npk.intvn-interval 7 --save-folder data/Wheat_Threshold_WK_Rand/ --env-reward RewardFertilizationThresholdWrapper --max-n 20 --max-p 20 --max-k 20 --max-w 20
