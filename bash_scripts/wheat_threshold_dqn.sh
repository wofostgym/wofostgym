#!/bin/bash

python3 train_agent.py --agent-type DQN --DQN.wandb-project-name npk_threshold --DQN.total-timesteps 2000000 --npk.random-reset --track --save-folder paper_data/DQN_Wheat_Threshold/ --env-reward RewardFertilizationThresholdWrapper --max-n 20 --max-p 20 --max-k 20 --max-w 20
