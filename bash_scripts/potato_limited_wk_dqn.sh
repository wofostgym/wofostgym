#!/bin/bash

python3 train_agent.py --agent-type DQN --DQN.wandb-project-name limited_wk_rand --npk.random-reset --npk.domain-rand --npk.scale 0.03 --track --npk.intvn-interval 7 --save-folder data/Potato_Limited_WK_Rand/ --env-reward RewardLimitedRunoffWrapper