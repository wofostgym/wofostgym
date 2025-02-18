#!/bin/bash

python3 train_agent.py --agent-type PPO --npk.random-reset --npk.intvn-interval 7 --PPO.wandb-project-name npk_wk --track --save-folder paper_data/PPO_Wheat_Week/
