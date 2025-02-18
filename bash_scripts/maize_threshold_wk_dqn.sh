#!/bin/bash

python3 train_agent.py --agent-type DQN --npk.random-reset --npk.domain-rand --npk.scale 0.03 --track --npk.intvn-interval 7  --DQN.wandb-project-name maize_threshold_wk_rand --agro-file maize_agro.yaml --save-folder data/Maize_Threshold_WK_Rand/ --env-reward RewardFertilizationThresholdWrapper --max-n 20 --max-p 20 --max-k 20 --max-w 10